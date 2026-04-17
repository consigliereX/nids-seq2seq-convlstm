from __future__ import annotations
from pathlib import Path
import json

from src.utils.config import load_yaml
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything
from src.utils.env import get_torch_env, python_env_summary

def main():
    cfg = load_yaml(Path("configs") / "base.yaml")
    seed = int(cfg.raw.get("seed", 42))
    deterministic = bool(cfg.raw.get("reproducibility", {}).get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    log_cfg = cfg.raw.get("logging", {})
    log_to_file = bool(log_cfg.get("log_to_file", True))
    log_dir = Path(log_cfg.get("log_dir", "results/logs"))
    log_file = str(log_dir / "phase0_env_check.log") if log_to_file else None
    logger = setup_logger("PHASE0", level=str(log_cfg.get("level", "INFO")), log_file=log_file)

    logger.info("=== ENV CHECK ===")
    logger.info(python_env_summary())

    tenv = get_torch_env()
    logger.info(f"Torch={tenv.torch_version} | Device={tenv.device} | CUDA_Available={tenv.cuda_available}")
    if tenv.cuda_available:
        logger.info(f"GPU={tenv.gpu_name} | CUDA={tenv.cuda_version} | cuDNN={tenv.cudnn_version}")

    # Write machine-readable snapshot (used later in Proof Package)
    out = {
        "python": python_env_summary(),
        "torch": tenv.__dict__,
        "seed": seed,
        "deterministic": deterministic,
    }
    Path("results").mkdir(exist_ok=True, parents=True)
    snap_path = Path("results") / "phase0_env_snapshot.json"
    snap_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info(f"Wrote: {snap_path.as_posix()}")

if __name__ == "__main__":
    main()
