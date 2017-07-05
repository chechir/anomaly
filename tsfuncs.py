from functools import partial
import numpy as np
import pandas as pd
from pandas import datetime
from pandas.tools.plotting import autocorrelation_plot

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
import seamless as ss

cutoff = '2000-05-05'


def lagged_rolling_mean(v, window):
    result = ss.np.rolling_mean(v, window)
    result = ss.np.lag(result, init=0)
    return result

def grouped_lagged_rolling_mean(v, window, groupby):
    func = partial(lagged_rolling_mean, window=window)
    result = ss.np.group_apply(v, groupby, func)
    return result

def ema(v, alpha=0.2):
    result = np.nan * v
    result[0] = v[0]
    for i in range(1, len(v)):
        result[i] = alpha * v[i] + (1 - alpha) * result[i-1]
    return result

def lagged_ema(v, alpha):
    emas = ema(v, alpha)
    emas = ss.np.lag(emas, init=0)
    return emas

def grouped_lagged_ema(v, alpha, groupby):
    func = partial(lagged_ema, alpha=alpha)
    result = ss.np.group_apply(v, groupby, func)
    return result

def dema(v, span, beta):
    intercept = v * np.nan
    slope = v * np.nan
    intercept[0] = v[0]
    slope[0] = 0
    alpha = 2.0 / (1 + span)
    for i in range(1, len(v)):
        intercept[i] = alpha * v[i] + (1 - alpha) * (intercept[i-1] + slope[i-1])
        slope[i] = (beta * (intercept[i] - intercept[i-1]) + (1 - beta) * slope[i-1])
    return intercept

def lagged_dema(v, span, beta):
    demas = dema(v, span, beta)
    demas = ss.np.lag(demas, init=0)
    return demas

def grouped_lagged_dema(v, span, beta, groupby):
    func = partial(lagged_dema, span=span, beta=beta)
    result = ss.np.group_apply(v, groupby, func)
    return result

def mape(actual, estimate):
    pcterrors = []
    for i in range(len(estimate)):
        pcterrors.append(abs(estimate[i]-actual[i])/actual[i])
    return sum(pcterrors)/len(pcterrors)

def test_stationarity(timeseries, window=12):
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=window)
    rolstd = pd.rolling_std(timeseries, window=window)

    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')

    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    plt.show()

def lagged_values(df, shift, fillna=0):
    values = ss.np.lag(df.values)
    values[~np.isfinite(values)] = fillna
    return values

def _parse_dates(x):
    result = datetime.strptime('190'+x, '%Y-%m')
    return result

def load_data(file_path):
    series = pd.read_csv(file_path, header=0, index_col=0, squeeze=True)
    series.index = [_parse_dates(x) for x in series.index.values]
    return series
    print series.head()

def get_arima_model(df):
    model = ARIMA(df, order=(4,1,0))
    return model

def fit_model(model):
    # fit model
    # This sets the lag value to 5 for autoregression,
    # uses a difference order of 1 to make the time series stationary, 
    # and uses a moving average model of 0.
    model_fit = model.fit(disp=0)
    return model_fit

def get_autocorrelation_plot(series, title = 'Autocorrelation plot'):
    autocorrelation_plot(series)
    plt.title(title)
    plt.show()
    # The horizontal lines displayed in the plot correspond to 95% and 99% confidence bands. The dashed line is 99% confidence band.

def get_kde_plot(df, title='Kernel Density Estimation'):
    df.plot(kind='kde')
    plt.title(title)
    plt.show()

def get_ts_plot(df, title='Time series plot'):
    df.plot()
    plt.title(title)
    plt.ylabel('Sales')
    plt.show()

def get_actual_vs_prediction_plot(actual, predictions):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(actual, color='g', label='Actual')
    ax.plot(predictions, color='c', label='Predictions')
    plt.legend()
    plt.ylabel('Value')
    plt.show()

def get_ixs(df, date=None, cutoff = cutoff):
    if date is not None:
        df.index = date
    ixs = df.index < cutoff
    return ixs

def get_rolling_predictions(model, train, val):
    history = [x for x in train]
    predictions = []
    for t in range(len(val)):
        model = get_arima_model(history)
        model_fit = fit_model(model)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = val[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    return predictions

