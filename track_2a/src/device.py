"""Pick the PyTorch device: CUDA on an NVIDIA GPU (Windows, Linux), MPS on a Mac with Apple Silicon, else the CPU.

    uv run python -m src.device     # show which device this machine uses, and try it
"""

import sys

import torch


def pick() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(device)}, CUDA {torch.version.cuda}"
    if device.type == "mps":
        return "Apple GPU via Metal (MPS)"
    if torch.version.cuda is None and sys.platform != "darwin":
        return "no GPU: this torch is a CPU-only build"
    return "no GPU found"


if __name__ == "__main__":
    device = pick()
    print(f"torch {torch.__version__} on {device}: {describe(device)}")
    x = torch.randn(1024, 1024, device=device)
    print(f"A matrix product on {device} works: {(x @ x).shape}")
