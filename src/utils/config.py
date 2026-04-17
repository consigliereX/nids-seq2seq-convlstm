from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]

    def get(self, key: str, default=None):
        return self.raw.get(key, default)

def load_yaml(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping. Got: {type(raw)}")
    return Config(raw=raw)
