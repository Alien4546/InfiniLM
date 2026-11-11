#include <infinicore/context/context.hpp>
#include <infinicore/ops/mha.hpp>
#include <hip/hip_runtime.h>
#include <ninetoothed.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {

using infinicore::DataType;
using infinicore::Device;
using infinicore::Tensor;
using NT = NineToothedTensor;
using Launch = NineToothedResult (*)(NineToothedStream, NT, NT, NT, NT, NT, NT, NT, NT, NT);

struct Kernel {
    void *handle = nullptr;
    Launch launch = nullptr;
};

std::array<Kernel, 2> kernels;
std::once_flag registration;
thread_local std::string last_error;

struct Plan {
    infinicore::graph::GraphTensor out, q, k, v;
    double scale;
    int kernel;
};

void validate(const Tensor &tensor) {
    if (tensor->ndim() != 4 || tensor->dtype() != DataType::BF16 || tensor->device().getType() != Device::Type::HYGON) {
        throw std::runtime_error("NineToothed MHA requires 4D HYGON BF16 tensors");
    }
    if (tensor->size(0) != 1 || tensor->size(1) == 0 || tensor->size(2) == 0 || tensor->strides()[3] != 1) {
        throw std::runtime_error("NineToothed MHA requires batch=1, positive sequence/heads and inner stride=1");
    }
    for (auto stride : tensor->strides()) {
        if (stride < 0) {
            throw std::runtime_error("NineToothed MHA does not support negative strides");
        }
    }
    if ((reinterpret_cast<std::uintptr_t>(tensor->data()) & 15) != 0) {
        throw std::runtime_error("NineToothed MHA requires 16-byte aligned tensor data");
    }
}

void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
           std::optional<Tensor> alibi, float scale, bool causal) {
    if (alibi.has_value() || causal) {
        throw std::runtime_error("NineToothed MHA supports noncausal attention without ALiBi only");
    }
    if (infinicore::context::isGraphRecording()) {
        throw std::runtime_error("NineToothed MHA requires eager inference");
    }
    validate(out);
    validate(q);
    validate(k);
    validate(v);
    if (out->shape() != q->shape() || k->shape() != q->shape() || v->shape() != q->shape()
        || !(out->device() == q->device()) || !(k->device() == q->device()) || !(v->device() == q->device())) {
        throw std::runtime_error("NineToothed MHA requires same-shape self-attention tensors on one device");
    }
    if (!std::isfinite(scale)) {
        throw std::runtime_error("NineToothed MHA scale must be finite");
    }
    const auto dim = q->size(3);
    const int index = dim == 72 ? 0 : dim == 128 ? 1 : -1;
    if (index < 0) {
        throw std::runtime_error("NineToothed MHA supports head dimensions 72 and 128 only");
    }
    if (out->data() == q->data() || out->data() == k->data() || out->data() == v->data()) {
        throw std::runtime_error("NineToothed MHA output must not alias its inputs");
    }
    return new Plan{infinicore::graph::GraphTensor(out), infinicore::graph::GraphTensor(q),
                    infinicore::graph::GraphTensor(k), infinicore::graph::GraphTensor(v),
                    static_cast<double>(scale), index};
}

struct Packed {
    std::array<std::uint64_t, 4> shape;
    std::array<std::int64_t, 4> strides;
    NT arg;

    explicit Packed(const Tensor &tensor) {
        // Core B,S,H,D -> NineToothed B,H,S,D without a data copy.
        constexpr int order[] = {0, 2, 1, 3};
        for (int i = 0; i < 4; ++i) {
            shape[i] = tensor->size(order[i]);
            strides[i] = tensor->strides()[order[i]];
        }
        arg = NT{const_cast<void *>(static_cast<const void *>(tensor->data())), shape.data(), strides.data()};
    }
};

void run(void *opaque) {
    auto *p = static_cast<Plan *>(opaque);
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device != p->q->device().getIndex()) {
        throw std::runtime_error("NineToothed MHA HIP device does not match the input tensor device");
    }
    Packed q(p->q), k(p->k), v(p->v), out(p->out);
    // The unmasked kernel still takes a descriptor for the unused mask.
    std::array<std::uint64_t, 4> mask_shape{1, 1, 1, 1};
    std::array<std::int64_t, 4> mask_strides{1, 1, 1, 1};
    NT mask{q.arg.data, mask_shape.data(), mask_strides.data()};
    std::int64_t zero = 0, one = 1;
    NT c0{&zero, nullptr, nullptr}, c1{&one, nullptr, nullptr};
    NT scale{&p->scale, nullptr, nullptr};
    auto stream = reinterpret_cast<NineToothedStream>(infinicore::context::getStream());
    const auto status = kernels[p->kernel].launch(stream, q.arg, k.arg, v.arg, mask, c0, scale, out.arg, c0, c1);
    if (status != 0) {
        throw std::runtime_error("NineToothed MHA launch failed: " + std::to_string(status));
    }
}

void cleanup(void **opaque) {
    delete static_cast<Plan *>(*opaque);
    *opaque = nullptr;
}

void install() {
    using Op = infinicore::op::MultiheadAttention;
    constexpr auto device = Device::Type::HYGON;
    if (Op::plan_dispatcher().lookup(device) || Op::run_dispatcher().lookup(device)
        || Op::cleanup_dispatcher().lookup(device)) {
        throw std::runtime_error("A HYGON MHA backend is already registered");
    }

    constexpr const char *path_vars[] = {"SONGSHAN_NT_MHA_D72_LIBRARY", "SONGSHAN_NT_MHA_D128_LIBRARY"};
    constexpr const char *symbols[] = {"launch_h3_nt_mha_bf16_d72_fp32math", "launch_h3_nt_mha_bf16_d128_bm32"};
    std::array<Kernel, 2> loaded;
    try {
        for (std::size_t i = 0; i < loaded.size(); ++i) {
            const auto *path = std::getenv(path_vars[i]);
            if (!path || !*path) {
                throw std::runtime_error(std::string("Missing NineToothed MHA library path: ") + path_vars[i]);
            }
            loaded[i].handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (!loaded[i].handle) {
                const auto *error = dlerror();
                throw std::runtime_error(std::string("NineToothed MHA dlopen failed: ") + (error ? error : path));
            }
            loaded[i].launch = reinterpret_cast<Launch>(dlsym(loaded[i].handle, symbols[i]));
            if (!loaded[i].launch) {
                throw std::runtime_error(std::string("NineToothed MHA missing symbol: ") + symbols[i]);
            }
        }
    } catch (...) {
        for (const auto &kernel : loaded) {
            if (kernel.handle) {
                dlclose(kernel.handle);
            }
        }
        throw;
    }

    // Keep the handles open for as long as the dispatcher holds our callbacks.
    kernels = loaded;
    Op::plan_dispatcher().registerDevice(device, &plan);
    Op::run_dispatcher().registerDevice(device, &run);
    Op::cleanup_dispatcher().registerDevice(device, &cleanup);
}

} // namespace

extern "C" int songshan_register_nt_mha_backend() noexcept {
    try {
        std::call_once(registration, install);
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return 1;
    }
}

extern "C" const char *songshan_nt_mha_last_error() noexcept {
    return last_error.c_str();
}
