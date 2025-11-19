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

