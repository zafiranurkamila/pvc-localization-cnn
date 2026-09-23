#!/usr/bin/env python3
import torch

print("="*70)
print("PyTorch 2.14.0 CUDA 13.2 - RTX 5060 Ti Blackwell Support Test")
print("="*70)
print()

print("Configuration:")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
print(f"  CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  GPU Memory: {props.total_memory / 1024**3:.2f} GB")
    print()

    print("Testing GPU Execution:")
    try:
        x = torch.randn(4000, 4000, device='cuda')
        y = torch.matmul(x, x.t())
        print("  Status: SUCCESS")
        print(f"  Tensor Device: {x.device}")
        print(f"  Matrix Multiply: WORKING")
        del x, y
        print()
        print("="*70)
        print("RESULT: RTX 5060 Ti Blackwell FULLY SUPPORTED!")
        print("GPU is ready for training!")
        print("="*70)
    except RuntimeError as e:
        if "no kernel image" in str(e):
            print("  Status: BLACKWELL NOT SUPPORTED")
        else:
            print(f"  Status: ERROR - {str(e)[:150]}")
else:
    print("CUDA not available")
