import numpy as np
import pandas as pd
from holy_hammer import data
from sklearn.metrics import mean_squared_error
from anomalias import tsfuncs as ts
import seamless as ss

target = 'meanp'
cutoff = '2000-01-01'

if __name__ == '__main__':
    df = data.load()
    #df = df.rowslice((df['horse_name'] == 'old_town_boy') | (df['horse_name'] == 'hayek'))
    #df = df.rowslice((df['horse_name'] == 'old_town_boy') )
    df[target] = df['place_flag']*0.6 + (1./df['place_odds'])*0.4

    df = df.to_pandas()
    df.index = df['scheduled_time']
    df = df[['horse_name',target]]

    tr, val = ts.split_data(df, cutoff)

    simple_exp = ts.grouped_lagged_ema(df[target].values, 0.70, df['horse_name'].values)
    tr, simple_exp_val = ts.split_data(pd.DataFrame(simple_exp), date=df.index, cutoff=cutoff)
    error = ts.mape(val[target].values, simple_exp_val.values)
    print('Simple exp MSE: %.3f' % error)

    #best mse:
    simple_exp = ts.grouped_lagged_ema(df[target].values, 0.99, df['horse_name'].values)
    tr, simple_exp_val = ts.split_data(pd.DataFrame(simple_exp), date=df.index, cutoff=cutoff)
    error = ts.mape(val[target].values, simple_exp_val.values)
    print('Simple exp MSE: %.3f' % error)

    #best mse:
    holt_exp = ts.grouped_lagged_dema(df[target].values, span=2, beta=0.10, groupby=df['horse_name'].values)
    tr, holt_exp_val = ts.split_data(pd.DataFrame(holt_exp), date=df.index, cutoff=cutoff)
    error = ts.mape(val[target].values, holt_exp_val.values)
    print('holt exp MSE: %.3f' % error)

    holt_exp = ts.grouped_lagged_dema(df[target].values, span=20, beta=0.05, groupby=df['horse_name'].values)
    tr, holt_exp_val = ts.split_data(pd.DataFrame(holt_exp), date=df.index, cutoff=cutoff)
    error = ts.mape(val[target].values, holt_exp_val.values)
    print('holt exp MSE: %.3f' % error)

    holt_exp = ts.grouped_lagged_dema(df[target].values, span=5, beta=0.50, groupby=df['horse_name'].values)
    tr, holt_exp_val = ts.split_data(pd.DataFrame(holt_exp), date=df.index, cutoff=cutoff)
    error = ts.mape(val[target].values, holt_exp_val.values)
    print('holt exp MSE: %.3f' % error)

    ts.get_actual_vs_prediction_plot(val[target].values, holt_exp_val.values)

