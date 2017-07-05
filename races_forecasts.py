import numpy as np
import pandas as pd
from holy_hammer import data
from horse.data import datasets
from sklearn.metrics import mean_squared_error
from anomalias import tsfuncs as ts
import seamless as ss

target = 'meanp'
cutoff = '2000-01-01'

def evaluate_ema(df, alpha):
    feature = ts.grouped_lagged_ema(df[target].values, alpha, df['horse_name'].values)
    feature_val = feature[val_ixs]
    error = np.sqrt(mean_squared_error(val[target].values, feature_val))
    print('EMA %.3f' % alpha + '- MSE: %.3f' % error)
    return feature_val

def evaluate_dema(df, span, beta):
    feature = ts.grouped_lagged_dema(df[target].values, span, beta, df['horse_name'].values)
    feature_val = feature[val_ixs]
    error = np.sqrt(mean_squared_error(val[target].values, feature_val))
    print('DEMA %.3f' % beta + '- MSE: %.3f' % error)
    return feature_val

def evaluate_rolling_mean(df, window):
    feature = ts.grouped_lagged_rolling_mean(df[target].values, window, df['horse_name'].values)
    feature_val = feature[val_ixs]
    error = np.sqrt(mean_squared_error(val[target].values, feature_val))
    print('Rolling mean %.3f' % window + '- MSE: %.3f' % error)
    return feature_val

if __name__ == '__main__':
    df = datasets.cleaned.load()
    #df = df.rowslice((df['horse_name'] == 'old_town_boy') | (df['horse_name'] == 'hayek'))
    #df = df.rowslice((df['horse_name'] == 'old_town_boy') )
    df[target] = df['place_flag']*0.6 + (1./df['place_odds'])*0.4

    df = df.to_pandas()
    df.index = df['scheduled_time']
    df = df[['horse_name',target]]

    ixs = df.index < cutoff
    val_ixs = ~ixs
    val = df.iloc[val_ixs]

    feature_val = evaluate_ema(df, 0.7)
    feature_val = evaluate_ema(df, 0.99)
    feature_val = evaluate_dema(df, 2, 0.1)
    feature_val = evaluate_dema(df, 20, 0.05)
    feature_val = evaluate_dema(df, 20, 0.5)

    ts.get_actual_vs_prediction_plot(val[target].values, feature_val)

