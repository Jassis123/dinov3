# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))

import huggingface_hub
print(huggingface_hub.__version__)

from huggingface_hub import get_full_repo_name
print(get_full_repo_name)