# 看进程实际用到哪一个 libnvJitLink.so.12
python - << 'PY'
import subprocess, sys
subprocess.run(
    "ldd $(python -c 'import torch, inspect; import torch._C; print(torch._C.__file__)') | grep nvJitLink",
    shell=True
)
PY
