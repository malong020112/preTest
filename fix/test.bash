python - << 'PY'
import torch, os, subprocess, textwrap
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH","<empty>"))
subprocess.run("ldd $(python -c 'import torch,inspect; \
import os; import torch; import ctypes; \
print(torch.__file__)') 2>/dev/null | grep nvJitLink || echo 'no nvJitLink in ldd'", shell=True)
PY
