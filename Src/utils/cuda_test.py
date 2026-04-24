# import torch

# print("CUDA disponibile:", torch.cuda.is_available())
# print("Numero device visibili:", torch.cuda.device_count())

# idx = 7

# if torch.cuda.is_available() and idx < torch.cuda.device_count():
#     print(f"cuda:{idx} esiste")
#     print("Nome:", torch.cuda.get_device_name(idx))
# else:
#     print(f"cuda:{idx} NON esiste")

import os
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count =", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))