import xgboost as xgb
import pandas as pd

def fit_xgb(X: pd.DataFrame, y: pd.Series, params: dict):
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model
