import numpy as np
import pandas as pd
from holy_hammer import data
from sklearn.metrics import mean_squared_error

from anomalias import tsfuncs as ts

if __name__ == '__main__':
    df = data.load()
    df = df.rowslice(df['horse_name']=='dvinsky')
    df['deltap'] = df['place_flag'] - (1./df['place_odds'])
    df['meanp'] = df['place_flag']*0.6 + (1./df['place_odds'])*0.4

    series = pd.DataFrame(df['meanp'].astype(float))
    series.columns = ['meanp']
    series.index = df['scheduled_time']

    ts.get_autocorrelation_plot(series, title='Residuals ACF, all should be whitin confidence interval')

    model = ts.get_arima_model(series)
    model_fit = ts.fit_model(model)
    print(model_fit.summary())

    residuals = pd.DataFrame(model_fit.resid)
    ts.get_autocorrelation_plot(residuals, title='Residuals ACF, all should be whitin confidence interval')
    #ts.get_kde_plot(residuals, title='Kernel Density Function for the residuals. Should be close to normal')

    train, val = ts.split_data(series)
    predictions = ts.get_rolling_predictions(model, train, val)

########################################################################
    # Evakuate
    error = mean_squared_error((val), np.exp(predictions))
    error = ts.mape((val), (predictions))
    print('ARIMA MSE: %.3f' % error)

    ts.get_actual_vs_prediction_plot((val), (predictions))

    pred_naive = series.shift(1)
    tr, pred_naive_val = ts.split_data(pred_naive)
    error = mean_squared_error((val), (pred_naive_val))
    error = ts.mape((val), (pred_naive_val))
    print('Naive MSE: %.3f' % error)

    pred_rolling = series.rolling(window=2).mean()
    pred_rolling = pred_rolling.shift(1)
    tr, pred_rolling_val = ts.split_data(pred_rolling)
    error = mean_squared_error((val), (pred_rolling_val))
    error = ts.mape((val), (pred_rolling_val))
    print('Rolling mean MSE: %.3f' % error)

    ewma_avg = series.ewm(halflife=5, ignore_na=False, min_periods=1, adjust=True).mean()
    tr, ewma_val = ts.split_data(pred_naive)
    error = mean_squared_error((val), (ewma_val))
    error = ts.mape((val), (ewma_val))
    print('EWMA MSE: %.3f' % error)

    simple_exp = ts.simple_exp_smoothing(series.values, alpha=0.45)
    tr, simple_exp_val = ts.split_data(pd.DataFrame(simple_exp))
    error = mean_squared_error((val), (simple_exp_val))
    error = ts.mape((val), (simple_exp_val))
    print('Simple exp MSE: %.3f' % error)

    holt_exp = ts.holt_exp_smoothing(series.values, span=3, beta=0.45)
    tr, holt_exp_val = ts.split_data(pd.DataFrame(holt_exp))
    error = mean_squared_error((val), (holt_exp_val))
    error = ts.mape(np.exp(val), (holt_exp_val))
    print('Simple exp MSE: %.3f' % error)

    holt_exp = ts.holt_exp_smoothing(series.values, span=20, beta=0.05)
    tr, holt_exp_val = ts.split_data(pd.DataFrame(holt_exp))
    error = mean_squared_error((val), (holt_exp_val))
    error = ts.mape(np.exp(val), (holt_exp_val))
    print('Simple exp MSE: %.3f' % error)
    ts.get_actual_vs_prediction_plot((val), (holt_exp_val))

