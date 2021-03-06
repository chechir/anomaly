import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from anomalias import tsfuncs as ts
import seamless as ss

file_path = ss.paths.dropbox() + 'files/sales-of-shampoo-over-a-three-ye.csv'
cutoff = '1902-03-01'


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

    train_ixs = ts.get_ixs(series, cutoff=cutoff)
    val_ixs = ~train_ixs
    train, val = series.iloc[train_ixs], series.iloc[~train_ixs]
    predictions = ts.get_rolling_predictions(model, train, val)

########################################################################
    # Evakuate
    error = mean_squared_error(np.exp(val), np.exp(predictions))
    error = ts.mape(np.exp(val), np.exp(predictions))
    print('ARIMA MSE: %.3f' % error)

    #ts.get_actual_vs_prediction_plot(np.exp(val), np.exp(predictions))

    pred_naive = series.shift(1)
    pred_naive_val = pred_naive[val_ixs]
    error = mean_squared_error(np.exp(val), np.exp(pred_naive_val))
    error = ts.mape(np.exp(val), np.exp(pred_naive_val))
    print('Naive MSE: %.3f' % error)

    pred_rolling = ts.lagged_rolling_mean(series.values, window=2)
    pred_rolling_val = pred_rolling[val_ixs]
    error = mean_squared_error(np.exp(val), np.exp(pred_rolling_val))
    error = ts.mape(np.exp(val), np.exp(pred_rolling_val))
    print('Rolling mean MSE: %.3f' % error)

    simple_exp = ts.lagged_ema(series, alpha=0.45)
    simple_exp_val = simple_exp[val_ixs]
    error = mean_squared_error(np.exp(val), np.exp(simple_exp_val.values))
    error = ts.mape(np.exp(val), np.exp(simple_exp_val.values))
    print('Simple exp MSE: %.3f' % error)

    ts.get_actual_vs_prediction_plot(val.values, simple_exp_val.values)
