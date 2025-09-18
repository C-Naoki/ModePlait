import subprocess
import sys

import torch
from tabulate import tabulate

# basic information
cuda_available = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
data = [
    ["Python Version", str(sys.version)],
    ["PyTorch Version", str(torch.__version__)],
    ["CUDA Version", str(torch.version.cuda)],
    ["CUDA Available", str(cuda_available)],
    ["cuDNN Version", str(torch.backends.cudnn.version())],
    ["Number of GPUs Available", str(gpu_count)],
]

print("\n" + "*" * 50)
print(" 1. Basic Information")
print("*" * 50)
print(tabulate(data, headers=["Item", "Value"], tablefmt="grid"))

print("\n" + "*" * 50)
print(" 2. Summary of GPUs")
print("*" * 50)
if cuda_available and gpu_count > 0:
    for i in range(gpu_count):
        memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(
            f"  - Memory Allocated: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB"
        )
        print(f"  - Memory Cached: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
        print(f"  - Memory Total: {memory_total:.2f} GB")
else:
    print("No GPUs available or CUDA is not enabled")

# device to use
device = torch.device("cuda" if cuda_available else "cpu")
print("Using device:", device)

print("\n" + "*" * 50)
print(" 3. Simple Test (Create Tensor on Device)")
print("*" * 50)
try:
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    print("Tensor created on device:", x.device)
    print("Tensor values:", x)
    print("Test completed! Let's start coding!")
except Exception as e:
    print("Error when creating tensor on device:", e)
    print("Please make sure that CUDA is properly installed and configured")

print("\n" + "*" * 50)
print(" 4. Additional Information")
print("*" * 50)

# additional information
try:
    driver_version = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        )
        .decode("utf-8")
        .strip()
    )
    print("CUDA Driver Version:", driver_version)
except Exception as e:
    print("Error getting CUDA Driver Version:", e)

try:
    toolkit_version = subprocess.check_output(
        [
            "nvcc",
            "--version",
        ]
    ).decode("utf-8")
    print("CUDA Toolkit Version:", toolkit_version)
except Exception as e:
    print("Error getting CUDA Toolkit Version:", e)
