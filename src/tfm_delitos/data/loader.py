from pathlib import Path
import pandas as pd
from typing import Dict

def load_main_dataset(raw_dir: str) -> pd.DataFrame:
    raw = Path(raw_dir)
    # Intenta cargar el principal si existe
    candidates = [
        "delitos_poblacion_semanal.csv",
        "enemu_semanal.csv",
        "ndd_datos.csv"
    ]
    for name in candidates:
        f = raw / name
        if f.exists():
            df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_excel(f)
            return df
    raise FileNotFoundError("No encontré archivos base en data/raw. Coloca tus CSV o XLSX con columnas esperadas.")
