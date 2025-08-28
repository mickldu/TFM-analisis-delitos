import argparse
import pandas as pd
from pathlib import Path
from tfm_delitos.config import load_config
from tfm_delitos.data.loader import load_main_dataset
from tfm_delitos.features.build_features import prepare_timeseries
from tfm_delitos.models.arimax import fit_arimax
from tfm_delitos.models.xgb import fit_xgb
from tfm_delitos.models.registries import save_model

def main(config_path: str):
    cfg = load_config(config_path)
    df = load_main_dataset(cfg.paths["raw_dir"])

    # Asume columnas con claves y target
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    keys = cfg.series_keys
    groups = df.groupby([keys["provincia"], keys["canton"], keys["delito"]])

    for (prov, cant, delit), g in groups:
        g = g.sort_values(cfg.date_col)
        y = g[cfg.target_col]
        exog_cols = [c for c in cfg.exogenous if c in g.columns]
        X = g[exog_cols] if exog_cols else None

        # ARIMAX
        try:
            arimax_res = fit_arimax(y, X)
            save_model(arimax_res, cfg.paths["registries_dir"], prov, cant, delit, "ARIMAX")
            print(f"ARIMAX entrenado para {prov}-{cant}-{delit}")
        except Exception as e:
            print(f"ARIMAX falló en {prov}-{cant}-{delit}: {e}")

        # XGBoost simple
        if X is not None and not X.empty:
            try:
                model = fit_xgb(X, y, cfg.xgboost)
                save_model(model, cfg.paths["registries_dir"], prov, cant, delit, "XGBOOST")
                print(f"XGBOOST entrenado para {prov}-{cant}-{delit}")
            except Exception as e:
                print(f"XGBOOST falló en {prov}-{cant}-{delit}: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
