# 看进程实际用到哪一个 libnvJitLink.so.12
python - << 'PY'
import subprocess, sys
subprocess.run(
    "ldd $(python -c 'import torch, inspect; import torch._C; print(torch._C.__file__)') | grep nvJitLink",
    shell=True
)
PY


#!/usr/bin/env bash
set -e

export CONDA_PREFIX=/opt/nas/p/conda/envs/verlaa

# 1. 明确把 torch 自带的 CUDA 库放在最前面
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\
${LD_LIBRARY_PATH:-}"

# 2. 可选：强制预加载正确的 nvJitLink
export LD_PRELOAD="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12:${LD_PRELOAD:-}"



#!/usr/bin/env bash
set -euo pipefail

# 你的 env 路径
export CONDA_PREFIX=/opt/nas/p/conda/envs/verlaa

echo ">>> 1. 查找 conda 里 nvjitlink 目录:"
python - << 'PY'
import nvidia.nvjitlink, os, inspect
print("nvidia.nvjitlink package dir:", os.path.dirname(inspect.getfile(nvidia.nvjitlink)))
PY

# 把上一段打印出的目录记一下，比如是：
# /opt/nas/p/conda/envs/verlaa/lib/python3.11/site-packages/nvidia/nvjitlink

NVJIT_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib"
CUSPARSE_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib"
CUBLAS_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib"
RUNTIME_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
NVRTC_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"

echo ">>> 2. 覆盖 LD_LIBRARY_PATH:"
export LD_LIBRARY_PATH="$NVJIT_DIR:$CUSPARSE_DIR:$CUBLAS_DIR:$RUNTIME_DIR:$NVRTC_DIR"

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo ">>> 3. 在同一个进程里检查 torch 和 nvJitLink:"
python - << 'PY'
import torch, os, subprocess
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("LD_LIBRARY_PATH in python:", os.environ.get("LD_LIBRARY_PATH","<empty>"))
subprocess.run(
    "ldd $(python -c 'import torch,inspect; import torch._C; print(torch._C.__file__)') | grep nvJitLink || echo NO_NVJIT",
    shell=True,
)
PY
