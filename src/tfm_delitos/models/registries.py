from pathlib import Path
import joblib
from typing import Tuple, Any

def key_to_path(registries_dir: str, provincia: str, canton: str, delito: str, model: str) -> Path:
    base = Path(registries_dir)
    base.mkdir(parents=True, exist_ok=True)
    name = f"{model}__{provincia}__{canton}__{delito}.joblib"
    return base / name

def save_model(model: Any, registries_dir: str, provincia: str, canton: str, delito: str, model_name: str):
    path = key_to_path(registries_dir, provincia, canton, delito, model_name)
    joblib.dump(model, path)

def load_model(registries_dir: str, provincia: str, canton: str, delito: str, model_name: str):
    path = key_to_path(registries_dir, provincia, canton, delito, model_name)
    if not path.exists():
        return None
    return joblib.load(path)
