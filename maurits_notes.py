
import numpy as np
import pandas as pd
from holy_hammer import data
from sklearn.metrics import mean_squared_error
from anomalias import tsfuncs as ts
import seamless as ss


if __name__ == '__main__':
    df = data.load()
    ixs = df.get_group_ixs('runner_id')
    ix = ixs.values()[100]
    distances = df['distance_yards'][ix]

    import matplotlib.pyplot as plt
    plt.plot(distances, label='dist')

    smoothed = ema(distances)
    plt.plot(smoothed, label='smoothed')

    smoothed = lagged_ema(distances, alpha=0.2, init=1800)
    plt.plot(smoothed, label='lagged')

    from functools import partial
    lagged_ema_func = partial(lagged_ema, alpha=0.2, init=1800)
    feature = ss.np.group_apply(df['distance_yards'], df['runner_id'], lagged_ema_func)

    plt.legend()
    plt.show()
    1/0
