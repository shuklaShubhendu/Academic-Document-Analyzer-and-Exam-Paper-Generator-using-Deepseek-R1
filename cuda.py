import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")