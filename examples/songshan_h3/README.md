# 嵩山杯 MiniMax-H3 FL2VA 复现

这组脚本为单节点 8 张海光 `gfx936` 构建 FL2VA。模型、首帧、提示词、输出路径和生成配置由调用方传入。源码构建包含 InfiniCore、InfiniLM、九齿生成的 D72/D128/D64 注意力和 FP32/BF16 CAST，以及 InfiniCore 的注意力注册器。

## 准备源码和依赖

在可访问 GitHub 的机器上，按报告中的公开链接和 commit 检出 `InfiniCore`、`InfiniLM`、`ninetoothed`。对 InfiniCore 和 InfiniLM 执行 `git submodule update --init --recursive`。构建器另外使用 [ntops](https://github.com/InfiniTensor/ntops) 的 `9ae4166ad342e4745f0eed13a5a20d069e994fc0`，检出该版本后将四个源码目录放到同一个可写工作区。

已测的组件为 DTK 26.04、Linux x86_64、Python 3.11、海光 PyTorch 2.9.0（HIP、C++ ABI=1）、海光 Triton 3.5.1、xmake 3.0.5 和 GCC/G++ 11.4。保留镜像提供的厂商 PyTorch 和基础包（含 SymPy 1.14.0）。把 `requirements-linux-cp311.txt` 中的 wheel 预置到本地目录。该文件固定了额外依赖和已验证的 wheel 哈希。

以下命令从四个源码目录的父目录运行。`WHEELHOUSE` 指向已准备好的离线 wheel 目录。

```bash
source /opt/dtk/env.sh
python3 -m venv --system-site-packages .h3-venv
source .h3-venv/bin/activate
python -m pip install --no-index --no-deps --require-hashes \
  --find-links "$WHEELHOUSE" \
  -r InfiniLM/examples/songshan_h3/requirements-linux-cp311.txt
```

## 在 CPU 工作区离线构建

`BUILD_DIR` 可以指向评测账号有写权限的持久目录。xmake 为 Core 和 LM 分别保存配置及中间文件。构建可能跨多个 CPU 工作区时长，重新进入同一环境后可用原来的 `BUILD_DIR` 继续该阶段。

```bash
BUILD_DIR="$PWD/h3-build"
BUILD_ARGS=(--core-dir "$PWD/InfiniCore" \
  --ninetoothed-dir "$PWD/ninetoothed" --ntops-dir "$PWD/ntops" \
  --build-dir "$BUILD_DIR" --dtk-root /opt/dtk --jobs 2)

python InfiniLM/examples/songshan_h3/build.py "${BUILD_ARGS[@]}" --stage check
```

环境检查输出 `H3_BUILD_ENVIRONMENT_OK` 后，依次执行以下四条命令。需要为 AOT 指定节点本地的临时目录时，在 `aot` 命令增加 `--scratch-dir`。

```bash
python InfiniLM/examples/songshan_h3/build.py "${BUILD_ARGS[@]}" --stage aot
python InfiniLM/examples/songshan_h3/build.py "${BUILD_ARGS[@]}" --stage core
python InfiniLM/examples/songshan_h3/build.py "${BUILD_ARGS[@]}" --stage lm
python InfiniLM/examples/songshan_h3/build.py "${BUILD_ARGS[@]}" --stage backend
```

最后一步写出 `$BUILD_DIR/manifest.json`，包含生成库的相对路径和 SHA256。Python 扩展安装在本次使用的 Core 和 LM 源码目录中。八卡作业需要能访问这两份源码、`BUILD_DIR` 和同一套 Python 依赖。

## 在八卡作业运行 FL2VA

加载同一套 DTK 和 Python 环境。`MODEL`、`FIRST_FRAME`、`PROMPT`、`OUTPUT` 由评测输入指定，生成参数以评测用例为准。下例演示已测规格。

```bash
python InfiniLM/examples/songshan_h3/run_fl2va.py \
  --build-dir "$BUILD_DIR" \
  --model-path "$MODEL" --first-frame "$FIRST_FRAME" \
  --text "$PROMPT" --output "$OUTPUT" \
  --gpus 0,1,2,3,4,5,6,7 --width 576 --height 384 \
  --num-frames 124 --steps 10 --seed 42
```

启动脚本使用生成清单校验 AOT 库，并把生成参数交给本仓库的 `examples/minimax_h3.py fl2va`。评测时检查进程退出码、视频可解码、分辨率、帧数、帧率和音视频流。对指定的参考输出，可使用本仓库的 `examples/minimax_h3_compare.py` 进行对比。
