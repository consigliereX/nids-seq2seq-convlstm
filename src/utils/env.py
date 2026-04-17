from __future__ import annotations
import platform
import sys
from dataclasses import dataclass

@dataclass(frozen=True)
class TorchEnv:
    torch_version: str | None
    cuda_available: bool
    device: str
    cuda_version: str | None
    cudnn_version: str | None
    gpu_name: str | None

def get_torch_env() -> TorchEnv:
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        device = "cuda" if cuda_available else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        cuda_version = torch.version.cuda if hasattr(torch.version, "cuda") else None
        cudnn_version = str(torch.backends.cudnn.version()) if hasattr(torch.backends, "cudnn") else None
        return TorchEnv(
            torch_version=torch.__version__,
            cuda_available=cuda_available,
            device=device,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            gpu_name=gpu_name,
        )
    except Exception:
        return TorchEnv(
            torch_version=None,
            cuda_available=False,
            device="cpu",
            cuda_version=None,
            cudnn_version=None,
            gpu_name=None,
        )

def python_env_summary() -> str:
    return f"Python={sys.version.split()[0]} | Platform={platform.platform()} | Exec={sys.executable}"
