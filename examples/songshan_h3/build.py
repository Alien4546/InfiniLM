"""Build the Songshan FL2VA candidate in an offline Linux DCU environment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
LM_DIR = HERE.parents[1]
TRITON_VERSION = "3.5.1+das.opt1.dtk2604.torch290"
AOT_PATHS = {
    "d72": "attention72/h3_nt_mha_bf16_d72_fp32math.so",
    "d128": "attention128/h3_nt_mha_bf16_d128_bm32.so",
    "d64": "attention64/h3_nt_mha_bf16_d64_fp32math.so",
    "cast": "cast/libsongshan_nt_cast.so",
}


def run(command, env, cwd):
    command = [str(arg) for arg in command]
    print(f"[{cwd}] {shlex.join(command)}", flush=True)
    subprocess.run(command, env=env, cwd=cwd, check=True)


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    return path


def digest(path):
    with require_file(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepend(env, name, *paths):
    inherited = env.get(name)
    env[name] = os.pathsep.join([*(str(path) for path in paths), *([inherited] if inherited else [])])


def configure_environment(args):
    if sys.platform != "linux" or sys.version_info[:2] != (3, 11):
        raise ValueError("Run this build with Python 3.11 in the Linux Hygon evaluation environment")
    for path in (
        args.core_dir / "xmake.lua",
        args.core_dir / "third_party/spdlog/include/spdlog/spdlog.h",
        LM_DIR / "third_party/spdlog/include/spdlog/spdlog.h",
        LM_DIR / "third_party/json/single_include/nlohmann/json.hpp",
        args.ninetoothed_dir / "src/ninetoothed/aot.py",
        args.ntops_dir / "src/ntops/kernels/scaled_dot_product_attention.py",
        args.dtk_root / "include/hip/hip_runtime.h",
        args.dtk_root / "lib/libamdhip64.so",
        args.dtk_root / "cuda/cuda/bin/nvcc",
    ):
        require_file(path)

    env = os.environ.copy()
    env.update(
        INFINI_ROOT=str(args.build_dir / "install"),
        HYGON_ARCH="gfx936",
        HYGON_CUDA_HOME=str(args.dtk_root / "cuda/cuda"),
        INFINILM_CXX11_ABI="1",
        XMAKE_GLOBALDIR=str(args.build_dir / "xmake-global"),
        PYTHON=sys.executable,
        PIP_NO_INDEX="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    prepend(env, "PATH", args.dtk_root / "cuda/cuda/bin", args.dtk_root / "bin")
    prepend(env, "LD_LIBRARY_PATH", args.build_dir / "install/lib", args.dtk_root / "lib", args.dtk_root / "cuda/cuda/lib64")
    prepend(env, "PYTHONPATH", args.core_dir / "python", LM_DIR / "python", args.ninetoothed_dir / "src", args.ntops_dir / "src")

    for program in ("xmake", "g++"):
        if not shutil.which(program, path=env["PATH"]):
            raise ValueError(f"Missing build tool in PATH: {program}")

    probe = """
import json
import sysconfig
from importlib import metadata
import pybind11
import torch

try:
    triton = metadata.version('triton')
except metadata.PackageNotFoundError:
    triton = None
print(json.dumps(dict(torch=str(torch.__version__), hip=torch.version.hip,
    abi=int(torch._C._GLIBCXX_USE_CXX11_ABI), triton=triton,
    pybind11=pybind11.get_include(), python_include=sysconfig.get_path('include'))))
"""
    result = subprocess.run([sys.executable, "-c", probe], env=env, check=True, text=True, stdout=subprocess.PIPE)
    info = json.loads(result.stdout.strip().splitlines()[-1])
    print(f"Build environment: {info}", flush=True)
    if info["torch"].split("+")[0] != "2.9.0" or not info["hip"] or info["abi"] != 1:
        raise ValueError("Expected the Hygon PyTorch 2.9.0 build with HIP and C++ ABI=1")
    if args.stage in ("check", "aot", "all") and info["triton"] != TRITON_VERSION:
        raise ValueError(f"Expected the tested Triton wheel {TRITON_VERSION}, found {info['triton']!r}")
    env["SONGSHAN_PYBIND11_INCLUDE"] = str(Path(info["pybind11"]).resolve())
    env["SONGSHAN_PYTHON_INCLUDE"] = str(Path(info["python_include"]).resolve())
    require_file(Path(env["SONGSHAN_PYBIND11_INCLUDE"]) / "pybind11/pybind11.h")
    require_file(Path(env["SONGSHAN_PYTHON_INCLUDE"]) / "Python.h")
    return env


def read_aot(build_dir):
    folder = build_dir / "aot"
    manifest = json.loads(require_file(folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["target"] != "gfx936":
        raise ValueError(f"Unexpected AOT target: {manifest['target']}")
    libraries = {}
    for name, expected_path in AOT_PATHS.items():
        entry = manifest["libraries"][name]
        if entry["path"] != expected_path or digest(folder / expected_path) != entry["sha256"]:
            raise ValueError(f"AOT artifact does not match its manifest: {name}")
        libraries[name] = {"path": f"aot/{expected_path}", "sha256": entry["sha256"]}
    header = require_file(folder / "cast/ninetoothed.h")
    if "cast_header_sha256" in manifest and digest(header) != manifest["cast_header_sha256"]:
        raise ValueError("CAST header does not match the AOT manifest")
    return libraries


def build_project(args, env, source, name, options, targets):
    work = args.build_dir / f"xmake-{name}"
    work.mkdir(parents=True, exist_ok=True)
    env = {**env, "XMAKE_CONFIGDIR": str(work)}
    project = ["-P", source, "-F", source / "xmake.lua"]
    # With -P/-F, xmake resolves a relative build directory from the working directory.
    run(["xmake", "f", *project, "-y", "-m", "release", "-o", "build", *options], env, work)
    for target in targets:
        run(["xmake", "build", *project, "-j", args.jobs, target], env, work)
        run(["xmake", "install", *project, "-y", target], env, work)


def build_backend(args, env):
    root = args.build_dir
    libraries = read_aot(root)
    prefix = root / "install"
    for name in ("infiniop", "infinirt", "infiniccl", "infinicore_cpp_api"):
        require_file(prefix / f"lib/lib{name}.so")
    for source, package, module in (
        (args.core_dir, "infinicore", "_infinicore"),
        (LM_DIR, "infinilm", "_infinilm"),
    ):
        folder = source / "python" / package / "lib"
        libraries_found = list(folder.glob(f"{module}*.so"))
        if len(libraries_found) != 1:
            raise ValueError(f"Expected one installed {module} extension in {folder}")
        require_file(libraries_found[0])

    relative_path = "install/lib/libsongshan_nt_mha.so"
    library = root / relative_path
    run([
        "g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread", "-Werror=return-type",
        "-D_GLIBCXX_USE_CXX11_ABI=1", "-D__HIP_PLATFORM_AMD__",
        f"-I{prefix / 'include'}", f"-I{args.dtk_root / 'include'}", f"-I{root / 'aot/cast'}",
        HERE / "nt_mha_backend.cpp",
        f"-L{prefix / 'lib'}", "-Wl,-rpath,$ORIGIN", "-linfinicore_cpp_api", "-linfinirt",
        f"-L{args.dtk_root / 'lib'}", f"-Wl,-rpath,{args.dtk_root / 'lib'}", "-lamdhip64",
        "-ldl", "-Wl,-z,defs", "-o", library,
    ], env, root)
    libraries["mha"] = {"path": relative_path, "sha256": digest(library)}
    manifest = dict(target="gfx936", core_dir=str(args.core_dir), lm_dir=str(LM_DIR),
                    dtk_root=str(args.dtk_root), libraries=libraries)
    output = root / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"FL2VA build manifest: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, required=True, help="InfiniCore checkout")
    parser.add_argument("--ninetoothed-dir", type=Path, required=True, help="NineToothed checkout")
    parser.add_argument("--ntops-dir", type=Path, required=True, help="ntops checkout at the revision in README.md")
    parser.add_argument("--build-dir", type=Path, required=True, help="Writable directory for build outputs")
    parser.add_argument("--dtk-root", type=Path, default=Path("/opt/dtk"))
    parser.add_argument("--scratch-dir", type=Path, help="Node-local scratch directory for AOT compilation")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--stage", choices=("check", "aot", "core", "lm", "backend", "all"), default="all")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    for name in ("core_dir", "ninetoothed_dir", "ntops_dir", "build_dir", "dtk_root", "scratch_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.build_dir in (args.core_dir, args.ninetoothed_dir, args.ntops_dir, LM_DIR):
        parser.error("Choose a build directory separate from the source directories")
    env = configure_environment(args)
    if args.stage == "check":
        print("H3_BUILD_ENVIRONMENT_OK", flush=True)
        return

    root = args.build_dir
    root.mkdir(parents=True, exist_ok=True)
    if args.stage in ("aot", "all"):
        command = [sys.executable, HERE / "build_aot.py", "--output-dir", root / "aot", "--dtk-root", args.dtk_root]
        if args.scratch_dir is not None:
            command += ["--scratch-dir", args.scratch_dir]
        run(command, env, Path.cwd())
        read_aot(root)
    if args.stage in ("core", "all"):
        read_aot(root)
        build_project(args, env, args.core_dir, "core", [
            "--songshan-offline=y", f"--songshan-nt-cast-dir={root / 'aot/cast'}",
            "--cpu=y", "--omp=y", "--hygon-dcu=y", "--hygon-arch=gfx936",
            f"--cuda={env['HYGON_CUDA_HOME']}", "--ninetoothed=n", "--ccl=y", "--aten=n",
            "--infiniops=n", "--cudnn=n",
        ], ("infinirt", "infiniccl", "infiniop", "infinicore_cpp_api", "_infinicore"))
    if args.stage in ("lm", "all"):
        require_file(root / "install/lib/libinfinicore_cpp_api.so")
        build_project(args, env, LM_DIR, "lm", ["--songshan-offline=y", "--cxx11-abi=1"], ("_infinilm",))
    if args.stage in ("backend", "all"):
        build_backend(args, env)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from None
