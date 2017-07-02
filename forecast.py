import numpy as np
import pandas as pd
from pandas import datetime
from pandas.tools.plotting import autocorrelation_plot

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import seamless as ss

file_path = ss.paths.dropbox() + 'files/sales-of-shampoo-over-a-three-ye.csv'

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

def load_data():
    series = pd.read_csv(file_path, header=0, index_col=0, squeeze=True)
    series.index = [_parse_dates(x) for x in series.index.values]
    return series
    print series.head()

def get_arima_model(df):
    model = ARIMA(df, order=(5,1,0))
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
    plt.plot(actual, color='g', label='Sales')
    plt.plot(predictions, color='c', label='Predictions')
    plt.legend()
    plt.ylabel('Sales')
    plt.show()

def split_data(df):
    X = df.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    return train, test

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

if __name__ == '__main__':
    series = load_data()
    get_ts_plot(series)
    get_autocorrelation_plot(series, 'Autocorrelation plot for the time series')
    test_stationarity(series)

    model = get_arima_model(series)
    model_fit = fit_model(model)
    print(model_fit.summary())

    residuals = pd.DataFrame(model_fit.resid)
    get_autocorrelation_plot(residuals, title='Residuals ACF, all should be whitin confidence interval')
    get_kde_plot(residuals, title='Kernel Density Function for the residuals. Should be close to normal')

    train, val = split_data(series)
    predictions = get_rolling_predictions(model, train, val)
    error = mean_squared_error(val, predictions)
    print('Validation MSE: %.3f' % error)

    # plot
    get_actual_vs_prediction_plot(val, predictions)

    pred_naive = series.shift(1)
    tr, pred_naive_val = split_data(pred_naive)

    pred_rolling = pd.rolling_mean(series, window=3)
    tr, pred_rolling_val = split_data(pred_rolling)
    error = mean_squared_error(val, pred_rolling_val)
    print('Validation MSE: %.3f' % error)
    get_actual_vs_prediction_plot(val, pred_rolling_val)

    ewma_avg = pd.ewma(series, halflife=12)
    tr, ewma_val = split_data(pred_naive)
    error = mean_squared_error(val, ewma_val)
    print('Validation MSE: %.3f' % error)

