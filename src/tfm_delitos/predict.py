import pandas as pd
from typing import Dict
from .models.registries import load_model
from .config import load_config

def predict_one(config_path: str, fecha: str, provincia: str, canton: str, delito: str, model_name: str="xgb") -> Dict:
    cfg = load_config(config_path)
    m = load_model(cfg.paths["registries_dir"], provincia, canton, delito, model_name.upper())
    if m is None:
        return {"status": "missing_model", "detail": f"No hay modelo entrenado para {provincia}-{canton}-{delito} con {model_name}."}
    # En XGBoost esperamos features exógenas. Aquí generamos un placeholder a cero
    # Ajusta esto para tomar las features reales de data/processed
    X = pd.DataFrame([[0]*1])
    pred = float(m.predict(X)[0])
    return {"status": "ok", "prediccion": pred, "fecha": fecha, "provincia": provincia, "canton": canton, "delito": delito, "modelo": model_name.upper()}
