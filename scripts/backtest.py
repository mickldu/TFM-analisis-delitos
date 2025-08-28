import argparse
import pandas as pd
from pathlib import Path
from tfm_delitos.config import load_config
from tfm_delitos.data.loader import load_main_dataset

def rolling_origin_backtest(df: pd.DataFrame, date_col: str, folds: int, horizon_weeks: int):
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    results = []
    step = max(1, n // (folds + 1))
    for i in range(folds):
        split = (i+1) * step
        train = df.iloc[:split]
        test = df.iloc[split: split + horizon_weeks]
        if test.empty:
            break
        results.append({
            "fold": i+1,
            "train_end": str(train[date_col].max()),
            "test_start": str(test[date_col].min()),
            "test_end": str(test[date_col].max()),
            "n_train": len(train),
            "n_test": len(test)
        })
    return pd.DataFrame(results)

def main(config_path: str, folds: int, horizon: int):
    cfg = load_config(config_path)
    df = load_main_dataset(cfg.paths["raw_dir"])
    keys = cfg.series_keys
    date_col = cfg.date_col
    # Ejemplo: solo una serie global
    report = rolling_origin_backtest(df, date_col, folds, horizon)
    out = Path("outputs")
    out.mkdir(exist_ok=True, parents=True)
    report.to_csv(out / "backtest_report.csv", index=False)
    print("Backtesting listo en outputs/backtest_report.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=1, help="Semanas")
    args = ap.parse_args()
    main(args.config, args.folds, args.horizon)
