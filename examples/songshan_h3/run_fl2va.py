"""Run InfiniLM FL2VA with the gfx936 NineToothed libraries from a local build."""

import argparse
import contextvars
import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import sysconfig
from time import perf_counter


_REEXEC_DIR = "_SONGSHAN_H3_REEXEC_DIR"
_AOT_LIBRARY_KEYS = ("d72", "d128", "d64", "cast", "mha")


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty build artifact: {path}")


def read_build_manifest(build_dir):
    manifest_path = build_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("target") != "gfx936":
        raise ValueError(f"{manifest_path} must describe a gfx936 build")

    source_dirs = {}
    for key in ("core_dir", "lm_dir", "dtk_root"):
        value = manifest.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing {key} in {manifest_path}")
        path = (build_dir / value).resolve()
        if not path.is_dir():
            raise ValueError(f"Directory for {key} not found: {path}")
        source_dirs[key] = path

    lm_dir = source_dirs["lm_dir"]
    if lm_dir != Path(__file__).resolve().parents[2]:
        raise ValueError(f"Run the FL2VA entry from the InfiniLM checkout built at {lm_dir}")
    require_file(lm_dir / "examples/minimax_h3.py")

    lib_dir = build_dir / "install/lib"
    for name in ("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl"):
        require_file(lib_dir / f"lib{name}.so")
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    for package, source_key in (("infinicore", "core_dir"), ("infinilm", "lm_dir")):
        require_file(source_dirs[source_key] / "python" / package / "lib" / f"_{package}{extension_suffix}")

    records = manifest.get("libraries")
    if not isinstance(records, dict):
        raise ValueError(f"Missing libraries in {manifest_path}")
    libraries = {}
    for key in _AOT_LIBRARY_KEYS:
        record = records.get(key)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"Missing library entry {key} in {manifest_path}")
        relative_path = Path(record["path"])
        path = (build_dir / relative_path).resolve()
        if relative_path.is_absolute() or not path.is_relative_to(build_dir):
            raise ValueError(f"Library {key} must be inside {build_dir}")
        require_file(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record.get("sha256"):
            raise ValueError(f"SHA256 mismatch for {key}: {path}")
        libraries[key] = path

    return source_dirs, libraries


def launch_with_library_paths(build_dir, source_dirs, libraries):
    env = os.environ.copy()
    install = build_dir / "install"
    dtk = source_dirs["dtk_root"]
    paths = {
        "LD_LIBRARY_PATH": (install / "lib", dtk / "lib", dtk / "cuda/cuda/lib64"),
        "PYTHONPATH": (source_dirs["core_dir"] / "python", source_dirs["lm_dir"] / "python"),
    }
    for name, prepend in paths.items():
        entries = [str(path) for path in prepend]
        if env.get(name):
            entries.append(env[name])
        env[name] = os.pathsep.join(entries)
    env.update(
        INFINI_ROOT=str(install),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        SONGSHAN_NT_CAST_LIBRARY=str(libraries["cast"]),
        SONGSHAN_NT_MHA_D72_LIBRARY=str(libraries["d72"]),
        SONGSHAN_NT_MHA_D128_LIBRARY=str(libraries["d128"]),
    )
    env[_REEXEC_DIR] = str(build_dir)
    env.pop("H3_STAGE_TRACE_DIR", None)
    # The native loader reads LD_LIBRARY_PATH at process startup.
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)


class TensorArg(ctypes.Structure):
    # Matches NineToothedTensor in the generated ninetoothed.h.
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_uint64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
    ]

    @classmethod
    def tensor(cls, tensor):
        shape = (ctypes.c_uint64 * tensor.ndim)(*tensor.shape)
        strides = (ctypes.c_int64 * tensor.ndim)(*tensor.stride())
        arg = cls(tensor.data_ptr(), shape, strides)
        arg._keep = (tensor, shape, strides)
        return arg

    @classmethod
    def scalar(cls, value, kind):
        cell = kind(value)
        arg = cls(ctypes.addressof(cell), None, None)
        arg._keep = cell
        return arg


def install_vae_attention(torch, vae_module, path):
    library = ctypes.CDLL(str(path))
    launch = library.launch_h3_nt_mha_bf16_d64_fp32math
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_void_p] + [TensorArg] * 9
    first_call = True

    def nt_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False,
                     backend=None, parallel_config=None, **kwargs):
        nonlocal first_call
        if attn_mask is not None or dropout_p or is_causal or enable_gqa or parallel_config is not None or kwargs:
            raise RuntimeError("NineToothed VAE attention requires unmasked noncausal self-attention")
        if not (query.shape == key.shape == value.shape and query.ndim == 4
                and query.shape[0] == 1 and query.shape[2:] == (32, 64)):
            raise RuntimeError(f"Unsupported VAE attention shapes: {query.shape}, {key.shape}, {value.shape}")
        for tensor in (query, key, value):
            if tensor.dtype != torch.bfloat16 or not tensor.is_cuda or tensor.device != query.device:
                raise RuntimeError(f"Unsupported VAE attention dtype/device: {tensor.dtype}, {tensor.device}")
            if tensor.stride(-1) != 1 or any(s <= 0 for s in tensor.stride()) or tensor.data_ptr() % 16:
                raise RuntimeError(f"Unsupported VAE attention strides/alignment: {tensor.stride()}")

        with torch.cuda.device(query.device):
            output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
            tensors = [tensor.transpose(1, 2) for tensor in (query, key, value)]
            # The unmasked kernel takes a valid descriptor for the unused mask.
            args = [TensorArg.tensor(tensor) for tensor in (*tensors, tensors[0])]
            args += [
                TensorArg.scalar(0, ctypes.c_int64),
                TensorArg.scalar(64 ** -0.5 if scale is None else scale, ctypes.c_double),
                TensorArg.tensor(output.transpose(1, 2)),
                TensorArg.scalar(0, ctypes.c_int64),
                TensorArg.scalar(1, ctypes.c_int64),
            ]
            stream = torch.cuda.current_stream(query.device)
            status = launch(ctypes.c_void_p(stream.cuda_stream), *args)
            if status:
                raise RuntimeError(f"NineToothed VAE attention launch failed: {status}")
        if first_call:
            print("VAE_NT_ATTENTION_FIRST_CALL:", tuple(query.shape), query.dtype, flush=True)
            first_call = False
        return output

    vae_module.dispatch_attention_fn = nt_attention
    print("VAE_NT_ADAPTER_ENABLED:", path, flush=True)


def install_h3_synchronization(torch, engine, runner, profile_enabled):
    active_step = contextvars.ContextVar("songshan_h3_step", default=None)
    steps = []
    original_call = runner.__call__
    original_input = runner._to_infinicore
    original_forward = engine.forward_raw

    def timed(function, key):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            row = active_step.get()
            if row is None or not profile_enabled:
                return function(*args, **kwargs)
            start = perf_counter()
            result = function(*args, **kwargs)
            row[key] = row.get(key, 0.0) + perf_counter() - start
            return result

        return wrapped

    wait = timed(torch.cuda.synchronize, "producer_wait_s")
    convert_input = timed(original_input, "input_wrap_s")
    submit_forward = timed(original_forward, "native_forward_host_s")

    @functools.wraps(original_input)
    def synchronized_input(tensor):
        if tensor.is_cuda:
            wait(tensor.device)
        return convert_input(tensor)

    @functools.wraps(original_call)
    def synchronized_call(self, *args, **kwargs):
        row = {"step": len(steps) + 1}
        token = active_step.set(row)
        start = perf_counter()
        try:
            result = original_call(self, *args, **kwargs)
            if profile_enabled:
                row["runner_wall_s"] = perf_counter() - start
                steps.append(row)
                print("H3_PROFILE_STEP", json.dumps(row), flush=True)
            return result
        finally:
            active_step.reset(token)

    @functools.wraps(original_forward)
    def forward_with_completion(self, *args, **kwargs):
        output = submit_forward(self, *args, **kwargs)
        row = active_step.get()
        if row is not None:
            start = perf_counter()
            # Complete native work while the caller still holds the inputs.
            for device in self.distributed_config.tp_device_ids:
                torch.cuda.synchronize(device)
            if profile_enabled:
                row["tp8_completion_wait_s"] = perf_counter() - start
        return output

    runner._to_infinicore = staticmethod(synchronized_input)
    runner._to_torch = staticmethod(timed(runner._to_torch, "output_bridge_s"))
    runner.__call__ = synchronized_call
    engine.forward_raw = forward_with_completion
    return steps


def generate(entry, forwarded, libraries, profile_path):
    import torch
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3
    from infinilm.infer_engine import InferEngine
    from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner

    steps = install_h3_synchronization(torch, InferEngine, MiniMaxH3TransformerRunner, profile_path is not None)
    install_vae_attention(torch, autoencoder_kl_minimax_h3, libraries["d64"])

    registrar = ctypes.CDLL(str(libraries["mha"]), mode=ctypes.RTLD_GLOBAL)
    register = registrar.songshan_register_nt_mha_backend
    register.argtypes = []
    register.restype = ctypes.c_int
    error = registrar.songshan_nt_mha_last_error
    error.argtypes = []
    error.restype = ctypes.c_char_p
    if register():
        raise RuntimeError((error() or b"NineToothed MHA registration failed").decode("utf-8", errors="replace"))
    print("NT_MHA_BACKEND_REGISTERED:", libraries["mha"], flush=True)

    if profile_path is not None:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [str(entry), "fl2va", *forwarded]
    try:
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        if profile_path is not None:
            profile_path.write_text(json.dumps({"steps": steps}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run InfiniLM FL2VA using the locally built gfx936 NineToothed backend.",
        epilog=(
            "FL2VA options are forwarded to examples/minimax_h3.py fl2va, including "
            "--model-path, --first-frame, --text, --output, --gpus, --width, --height, "
            "--num-frames, --steps and --seed. Put -- before the FL2VA options to "
            "forward its -h or --help."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--build-dir", type=Path, required=True, help="Directory containing the build manifest")
    parser.add_argument("--profile-json", type=Path, help="Save per-step timing to this file")
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    build_dir = args.build_dir.resolve()
    try:
        source_dirs, libraries = read_build_manifest(build_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if os.environ.pop(_REEXEC_DIR, None) != str(build_dir):
        launch_with_library_paths(build_dir, source_dirs, libraries)
    generate(source_dirs["lm_dir"] / "examples/minimax_h3.py", forwarded, libraries, args.profile_json)


if __name__ == "__main__":
    main()
