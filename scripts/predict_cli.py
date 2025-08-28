import argparse
from tfm_delitos.predict import predict_one

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fecha", required=True)
    ap.add_argument("--provincia", required=True)
    ap.add_argument("--canton", required=True)
    ap.add_argument("--delito", required=True)
    ap.add_argument("--modelo", default="xgb", choices=["xgb","arimax","tft"])
    args = ap.parse_args()
    res = predict_one(args.config, args.fecha, args.provincia, args.canton, args.delito, args.modelo)
    print(res)
