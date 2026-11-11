"""Build the Songshan H3 gfx936 AOT kernels offline, without a DCU."""

import argparse
import ast
import errno
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


TARGET = "gfx936"
ATTENTION_KERNELS = (
    (72, 16, "h3_nt_mha_bf16_d72_fp32math"),
    (128, 32, "h3_nt_mha_bf16_d128_bm32"),
    (64, 16, "h3_nt_mha_bf16_d64_fp32math"),
)
LIBRARIES = {
    f"d{head_dim}": f"attention{head_dim}/{kernel}.so"
    for head_dim, _, kernel in ATTENTION_KERNELS
}
LIBRARIES["cast"] = "cast/libsongshan_nt_cast.so"
CAST_HEADER = "cast/ninetoothed.h"


def replace_once(source, old, new, component):
    if source.count(old) != 1:
        raise RuntimeError(f"Unexpected {component} source. Expected one occurrence of {old!r}.")
    return source.replace(old, new, 1)


def private_cache_put(cache, data, filename, binary=True):
    """Keep the tested EXDEV fallback within this invocation's private cache."""
    destination = cache._make_path(filename)
    temporary = Path(cache._make_path(f".pending_{uuid.uuid4().hex}"))
    try:
        if isinstance(data, bytes):
            temporary.write_bytes(data)
        else:
            temporary.write_text(str(data), encoding="utf-8")
        try:
            os.replace(temporary, destination)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def configure_offline_toolchain(scratch):
    temporary = scratch / "tmp"
    temporary.mkdir()
    os.environ["TMPDIR"] = str(temporary)
    tempfile.tempdir = str(temporary)
    os.environ["TRITON_CACHE_DIR"] = str(scratch / "triton-cache")
    os.environ["NINETOOTHED_CACHE_DIR"] = str(scratch / "ninetoothed-cache")
    os.environ["NINETOOTHED_AOT_SKIP_LIBRARY"] = "1"

    import ninetoothed as nt
    from triton.backends.amd.compiler import HIPBackend
    from triton.backends.amd.driver import ty_to_cpp
    from triton.runtime.cache import FileCacheManager

    generation = importlib.import_module("ninetoothed.generation")
    aot = importlib.import_module("ninetoothed.aot")
    tool = importlib.import_module("triton.tools.compile")
    FileCacheManager.put = private_cache_put

    def offline_init(generator):
        ast.NodeTransformer.__init__(generator)
        generator._min_num_elements = 1
        generator._max_num_elements = 32768

    generation.CodeGenerator.__init__ = offline_init
    generation.CACHE_DIR = Path(os.environ["NINETOOTHED_CACHE_DIR"])
    generation.CACHE_DIR.mkdir(exist_ok=True)
    aot.calculate_default_configs = lambda: (4, 1)

    original_amdgcn = HIPBackend.make_amdgcn
    gep = re.compile(r"\bgetelementptr((?:\s+(?:inbounds|nuw|nusw))+)")

    def compatible_amdgcn(source, metadata, options):
        # The DTK LLVM reader accepts inbounds, but not the newer GEP wrap flags.
        source = gep.sub(
            lambda match: "getelementptr" + "".join(
                f" {flag}" for flag in match[1].split() if flag not in ("nuw", "nusw")
            ),
            source,
        )
        return original_amdgcn(source, metadata, options)

    HIPBackend.make_amdgcn = staticmethod(compatible_amdgcn)

    # Use the vendor HIP templates without querying a device for type mappings.
    source = replace_once(
        inspect.getsource(tool.compile_kernel),
        "triton.runtime.driver.active.map_python_to_cpp_type",
        "_hip_type_map",
        "Triton AOT compiler",
    )
    namespace = dict(vars(tool), _hip_type_map=ty_to_cpp)
    exec(compile(source, "<offline_triton_aot>", "exec"), namespace)
    compile_kernel = namespace["compile_kernel"]
    compiled = scratch / "compiled"
    compiled.mkdir()

    class ExplicitTarget:
        def split(self, _separator):
            return "hip", TARGET, 64

    def offline_compile(path, name, signature, grid, num_warps, num_stages):
        booleans = {"False": "0", "True": "1"}
        tokens = (value.strip() for value in signature.split(","))
        signature = ", ".join(booleans.get(value, value) for value in tokens)
        args = tool.CompileArgs(
            path=str(path), kernel_name=name, signature=signature, grid=grid,
            target=ExplicitTarget(), num_warps=num_warps, num_stages=num_stages,
            out_path=compiled / name,
        )
        _, files = compile_kernel(args)
        if not files:
            raise RuntimeError(f"Triton produced no host adapter files for {name}.")
        signature_hash = files[0].name.split(".")[1]
        contents = {}
        for file in files:
            # NineToothed uses a .c template key for the vendor's HIP .cpp file.
            key = file.with_suffix(".c").name if file.suffix == ".cpp" else file.name
            contents[key] = re.sub(r"\bwarpSize\b", "64", file.read_text(encoding="utf-8"))
        return signature_hash, contents

    aot._compile = offline_compile
    aot._enumerate_variant_specs = lambda *_: [("generic_i64", (), (), nt.int64, nt.int64)]
    source = replace_once(
        inspect.getsource(aot._build_variant),
        'param_types.append(f"{stride_type}:16")',
        "param_types.append(stride_type)",
        "NineToothed stride specialization",
    )
    # Reuse the actual module globals so each kernel uses its chosen header path.
    exec(compile(source, "<unhinted_nt_aot>", "exec"), vars(aot))
    return nt, aot


def generate_kernel(nt, name, output, arrange, application, tensors, header_path):
    output.mkdir(parents=True, exist_ok=True)
    nt.make(
        arrange, application, tensors, caller="cuda", kernel_name=name,
        output_dir=output, num_warps=4, num_stages=1,
    )

    header = output / f"{name}.h"
    header_text = header.read_text(encoding="utf-8")
    signature = re.search(rf"NineToothedResult launch_{name}\((.*?)\);", header_text, re.S)
    if signature is None:
        raise RuntimeError(f"Cannot find launch_{name} in {header}.")
    params = signature[1]
    arguments = ", ".join(param.strip().split()[-1] for param in params.split(","))
    (output / f"{name}.cpp").write_text(
        f'#include "{name}.h"\n'
        f'extern "C" NineToothedResult launch_{name}_generic_i64({params});\n'
        f'extern "C" NineToothedResult launch_{name}({params}) {{\n'
        f"    return launch_{name}_generic_i64({arguments});\n"
        "}\n",
        encoding="utf-8",
    )

    variant = output / f"{name}.generic_i64.cpp"
    source = variant.read_text(encoding="utf-8")
    for old, new in (
        ("ThreadSafeUnorderedMap<unsigned long long,", "ThreadSafeUnorderedMap<std::string,"),
        ("unsigned long long ctx_id = 0;", "std::string ctx_id;"),
        (
            "ctx_id = (unsigned long long)(uintptr_t)ctx;",
            "int device_id = -1;\n"
            "    auto status = hipGetDevice(&device_id);\n"
            "    if (status != hipSuccess) return static_cast<int>(status);\n"
            '    ctx_id = std::to_string(device_id) + ":" + std::to_string((uintptr_t)ctx);',
        ),
    ):
        source = replace_once(source, old, new, f"{name} HIP cache")

    # Keep generated host sources independent of their temporary build location.
    absolute_include = f'#include "{header_path}"'
    relative_include = '#include "ninetoothed.h"'
    header.write_text(header_text.replace(absolute_include, relative_include), encoding="utf-8")
    variant.write_text(
        "#include <string>\n" + source.replace(absolute_include, relative_include),
        encoding="utf-8",
    )
    return [output / f"{name}.cpp", variant]


def load_dsl(path, source):
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_attention(nt, aot, output, head_dim, block_size_m, name):
    from ntops.kernels.scaled_dot_product_attention import premake

    output.mkdir()
    aot._HEADER_PATH = output / "ninetoothed.h"
    arrange, application, tensors = premake(
        False, emb_dim=head_dim, is_causal=False, with_attn_mask=False,
        causal_variant=1, dtype=nt.bfloat16, block_size_m=block_size_m, block_size_n=32,
    )
    # Preserve the FP64 scalar in the host ABI, but use FP32 inside the DSL.
    source = replace_once(
        inspect.getsource(application),
        "1.4426950408889634 * scale * query[i]",
        "1.4426950408889634 * ntl.cast(scale, ntl.float32) * query[i]",
        "ntops attention application",
    )
    module = load_dsl(
        output / "attention_fp32math_dsl.py", "import ninetoothed.language as ntl\n\n" + source,
    )
    application = getattr(module, application.__name__)
    return generate_kernel(nt, name, output, arrange, application, tensors, aot._HEADER_PATH)


def build_cast(nt, aot, output):
    output.mkdir()
    aot._HEADER_PATH = output / "ninetoothed.h"
    # CodeGenerator reads the DSL source and resolves ntl through module globals.
    module = load_dsl(output / "cast_dsl.py", """\
import ninetoothed.language as ntl


def arrange_cast(output, source):
    return output.tile((256,)), source.tile((256,))


def application_convert(output, source):
    output = ntl.cast(source, output.dtype)
""")
    sources = []
    for direction, output_dtype, input_dtype in (
        ("f32_bf16", nt.bfloat16, nt.float32),
        ("bf16_f32", nt.float32, nt.bfloat16),
    ):
        tensors = (nt.Tensor(1, dtype=output_dtype), nt.Tensor(1, dtype=input_dtype))
        sources.extend(generate_kernel(
            nt, f"h3_nt_cast_{direction}", output / direction,
            module.arrange_cast, module.application_convert, tensors, aot._HEADER_PATH,
        ))
    return sources


def link_library(sources, library, compiler, hip_include, hip_lib):
    command = [
        compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread",
        "-Werror=return-type", "-D__HIP_PLATFORM_AMD__",
        f"-I{hip_include}", f"-I{library.parent}", *map(str, sources),
        f"-L{hip_lib}", f"-Wl,-rpath,{hip_lib}", "-lamdhip64", "-o", str(library),
    ]
    print(f"Linking {library.name}", flush=True)
    subprocess.run(command, check=True, timeout=300)


def sha256(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_manifest(output):
    manifest = output / "manifest.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("target") != TARGET or set(data.get("libraries", ())) != set(LIBRARIES):
        raise ValueError(f"Unexpected target or libraries in {manifest}.")
    for key, relative in LIBRARIES.items():
        record = data["libraries"][key]
        if record.get("path") != relative or sha256(output / relative) != record.get("sha256"):
            raise ValueError(f"AOT library verification failed: {output / relative}.")
    if sha256(output / CAST_HEADER) != data.get("cast_header_sha256"):
        raise ValueError(f"AOT header verification failed: {output / CAST_HEADER}.")
    return data


def publish(staged, output):
    import fcntl

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".build.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if read_manifest(output) is not None:
            return
        # Publish the manifest last. An interrupted copy can be rerun safely.
        for relative in (*LIBRARIES.values(), CAST_HEADER, "manifest.json"):
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".pending-{uuid.uuid4().hex}")
            try:
                shutil.copy(staged / relative, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)


def build(output, dtk_root, scratch_parent):
    if read_manifest(output) is not None:
        print(f"Verified existing gfx936 AOT libraries in {output}", flush=True)
        return
    if sys.platform != "linux":
        raise RuntimeError("AOT compilation requires the Linux Hygon DTK environment.")
    hip_include = dtk_root / "include"
    if not (hip_include / "hip/hip_runtime.h").is_file():
        raise FileNotFoundError(f"HIP headers missing under {hip_include}.")
    hip_lib = next(
        (dtk_root / name for name in ("lib", "lib64") if (dtk_root / name / "libamdhip64.so").is_file()),
        None,
    )
    if hip_lib is None:
        raise FileNotFoundError(f"libamdhip64.so missing under {dtk_root}/lib or lib64.")
    compiler = shutil.which("g++")
    if compiler is None:
        raise FileNotFoundError("g++ is required to link the HIP host adapters.")

    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="songshan-h3-aot-", dir=scratch_parent) as directory:
        scratch = Path(directory)
        nt, aot = configure_offline_toolchain(scratch)
        staged = scratch / "artifacts"
        staged.mkdir()
        for head_dim, block_size_m, name in ATTENTION_KERNELS:
            library = staged / LIBRARIES[f"d{head_dim}"]
            sources = build_attention(nt, aot, library.parent, head_dim, block_size_m, name)
            link_library(sources, library, compiler, hip_include, hip_lib)

        library = staged / LIBRARIES["cast"]
        sources = build_cast(nt, aot, library.parent)
        link_library(sources, library, compiler, hip_include, hip_lib)

        data = {
            "target": TARGET,
            "libraries": {
                key: {"path": relative, "sha256": sha256(staged / relative)}
                for key, relative in LIBRARIES.items()
            },
            "cast_header_sha256": sha256(staged / CAST_HEADER),
        }
        (staged / "manifest.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        publish(staged, output)
    read_manifest(output)
    print(f"Built and verified gfx936 AOT libraries in {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="AOT artifact directory")
    parser.add_argument("--dtk-root", type=Path, default=Path("/opt/dtk"), help="Hygon DTK root")
    parser.add_argument("--scratch-dir", type=Path, help="Parent of the private build/cache directory")
    args = parser.parse_args()
    scratch = args.scratch_dir.resolve() if args.scratch_dir is not None else None
    build(args.output_dir.resolve(), args.dtk_root.resolve(), scratch)


if __name__ == "__main__":
    main()
