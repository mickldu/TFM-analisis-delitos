import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_arimax(y: pd.Series, exog: pd.DataFrame, order=(1,1,1), seasonal_order=(0,0,0,0)):
    model = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res
