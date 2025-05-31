import torch
from torch.utils import cpp_extension

print(torch.version.cuda)               # CUDA版本号
print(cpp_extension.CUDA_HOME)          # CUDA安装路径  /usr/local/cuda

import tinycudann as tc
print(tc.doc)
import nerfacc
print(nerfacc._C)   # 应输出一个 <module ...>，而非 None

