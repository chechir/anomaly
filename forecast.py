import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from anomalias import tsfuncs as ts
import seamless as ss

file_path = ss.paths.dropbox() + 'files/sales-of-shampoo-over-a-three-ye.csv'


if __name__ == '__main__':
    series = ts.load_data(file_path)
    series = np.log(series)
    #ts.get_ts_plot(series)
    #ts.get_autocorrelation_plot(series, 'Autocorrelation plot for the time series')
    #ts.test_stationarity(series)

    model = ts.get_arima_model(series)
    model_fit = ts.fit_model(model)
    print(model_fit.summary())

    residuals = pd.DataFrame(model_fit.resid)
    #ts.get_autocorrelation_plot(residuals, title='Residuals ACF, all should be whitin confidence interval')
    #ts.get_kde_plot(residuals, title='Kernel Density Function for the residuals. Should be close to normal')

    train, val = ts.split_data(series)
    predictions = ts.get_rolling_predictions(model, train, val)

########################################################################
    # Evakuate
    error = mean_squared_error(np.exp(val), np.exp(predictions))
    error = ts.mape(np.exp(val), np.exp(predictions))
    print('ARIMA MSE: %.3f' % error)

    #ts.get_actual_vs_prediction_plot(np.exp(val), np.exp(predictions))

    pred_naive = series.shift(1)
    tr, pred_naive_val = ts.split_data(pred_naive)
    error = mean_squared_error(np.exp(val), np.exp(pred_naive_val))
    error = ts.mape(np.exp(val), np.exp(pred_naive_val))
    print('Naive MSE: %.3f' % error)

    pred_rolling = series.rolling(window=2).mean()
    pred_rolling = pred_rolling.shift(1)
    tr, pred_rolling_val = ts.split_data(pred_rolling)
    error = mean_squared_error(np.exp(val), np.exp(pred_rolling_val))
    error = ts.mape(np.exp(val), np.exp(pred_rolling_val))
    print('Rolling mean MSE: %.3f' % error)

    ewma_avg = series.ewm(halflife=5, ignore_na=False, min_periods=1, adjust=True).mean()
    tr, ewma_val = ts.split_data(pred_naive)
    error = mean_squared_error(np.exp(val), np.exp(ewma_val))
    error = ts.mape(np.exp(val), np.exp(ewma_val))
    print('EWMA MSE: %.3f' % error)

    simple_exp = ts.simple_exp_smoothing(series, alpha=0.45)
    tr, simple_exp_val = ts.split_data(pd.DataFrame(simple_exp))
    error = mean_squared_error(np.exp(val), np.exp(simple_exp_val))
    error = ts.mape(np.exp(val), np.exp(simple_exp_val))
    print('Simple exp MSE: %.3f' % error)

