import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Config:
    seed: int
    paths: Dict[str, str]
    series_keys: Dict[str, str]
    date_col: str
    target_col: str
    exogenous: List[str]
    frequency: str
    tft: Dict[str, Any]
    xgboost: Dict[str, Any]
    arimax: Dict[str, Any]
    backtest: Dict[str, Any]

def load_config(path: str) -> Config:
    p = Path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Config(**cfg)
