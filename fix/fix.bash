export CONDA_PREFIX=/opt/nas/p/conda/envs/verlaa

# 1. 把 torch 带的 nvidia 库放在最前面
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\
${LD_LIBRARY_PATH:-}"

# 2. 可选：确保预加载的是正确的 nvJitLink
export LD_PRELOAD="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12:${LD_PRELOAD:-}"
