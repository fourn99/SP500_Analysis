#%%
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model

import matplotlib as mpl


from preprocessing_data import *


#%%
# -- get processed data from script
df_data = df_time_series.copy()


# %%
# -- Stationary Test DF
from statsmodels.tsa.stattools import adfuller


def ts_stationarity_test(time_serie, name_series):

    moving_avg = time_serie.rolling(window=12).mean()
    moving_std = time_serie.rolling(window=12).std()
    orig = plt.plot(time_serie, color='blue', label='Original')
    mean = plt.plot(moving_avg, color='red', label='Rolling Mean 12M')
    std = plt.plot(moving_std, color='black', label='Rolling Stand. Dev 12M')
    plt.legend(loc='best')
    plt.show(block=False)

    plt.clf()
    plot_acf(time_serie)
    plt.title('ACF')
    plt.show()

    # c = constant, ct = constant + trend , ctt =  constant, and linear and quadratic trend., nc = no constant}
    regressions = ["nc", "c", "ct"]
    for index, value in enumerate(regressions):
        print('Results Dickey Fuller Test - ' + value)
        regression_type = value
        df_test_DF = adfuller(time_serie, autolag='AIC', regression=regression_type)
        df_output_results = pd.Series(df_test_DF[0:4], index=['Test Statistics', 'p-value', '#Lags Used', 'Number of Obs Used'])

        for key, value in df_test_DF[4].items():

            df_output_results['Criticial Value (%s)' % key] = value

        # data for bar plot of results
        x = [df_output_results.index[0], df_output_results.index[4], df_output_results.index[5], df_output_results.index[6]]
        height = [df_output_results[0], df_output_results[4], df_output_results[5], df_output_results[6]]

        plt.clf()
        plt.barh(x, height)
        plt.title(name_series +' ADF Test ' + regression_type + ' - P-Value: ' + str(round(df_output_results[1], 5)))
        for index, value in enumerate(height):
            plt.text(value, index, str(round(value, 3)))

        plt.show()
        print(df_output_results)
        print('\n')


# %%
ts_stationarity_test(df_data.SP500, "Price")

'''
Results: non-stationary all around
'''

# %%
log_price = np.log(df_data.SP500)
log_price.dropna(inplace=True)
ts_stationarity_test(log_price, "Log Price")  # ADF on Log Price

'''
Results indicate non-stationary for No constant, notrend; constants
Stationary with constant and with trend
'''

# %%

df_results_adf = ts_stationarity_test(df_data.returns, "Returns")  # ADF on Returns

'''
Results indicates stationary for all regression types
'''

# %%
log_returns = np.log(df_data.returns)
log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
log_returns.dropna(inplace=True)

ts_stationarity_test(log_returns, "Log Returns")  # ADF on Log Returns

'''
Results: non-stationary no constant/no trend
stationary for constant, no trend; constant and trend;
'''

# %%

# log_returns = np.log(df_data.returns)
# log_returns.dropna(inplace=True)
# df_results_adf = ts_stationarity_test(log_returns)  # ADF on Log Returns

# movingAverage = log_returns.rolling(window=12).mean()
# movingSTD = log_returns.rolling(window=12).std()
# log_returns_minus_ma = log_returns - movingAverage
# log_returns_minus_ma.dropna(inplace=True)


# ts_stationarity_test(log_returns_minus_ma)

# expo_decay = log_returns_minus_ma.ewm(halflife=12, min_periods=0, adjust=True).mean()
# plt.plot(log_returns_minus_ma)
# plt.plot(expo_decay, color='red')
# plt.title('Log Price - MA vs Expo Decay')
# plt.show()
# ts_stationarity_test(expo_decay)
#
# log_returns_minus_expdecay = log_returns - expo_decay
# ts_stationarity_test(log_returns_minus_expdecay)
# plt.plot(log_returns_minus_expdecay)
# plt.title('Log price - Weighted Avg Exponential decay')
# plt.show()
# ts_stationarity_test(log_returns_minus_expdecay)
#
# log_returns_minus_delay = log_returns - log_returns.shift()
# log_returns_minus_delay.dropna(inplace=True)
# plt.plot(log_returns_minus_delay, color='red')
# plt.title('Log Price - Log Price Shift')
# plt.show()
# ts_stationarity_test(log_returns_minus_delay)

# from statsmodels.tsa.seasonal import seasonal_decompose
# # log_returns.dropna(inplace=True)
# # print(log_returns.isna())
# decomposition = seasonal_decompose(log_returns, freq=12)
#
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residuals = decomposition.resid
#
# plt.subplot(411)
# plt.plot(log_returns, label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal, label='Seasonal')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residuals, label='Residuals')
# plt.legend(loc='best')
# plt.tight_layout()
#
# decomposedData = residuals
# decomposedData.dropna(inplace=True)
# ts_stationarity_test(decomposedData)
#%%
def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
        plt.show()
    return
log_price = np.log(df_data.SP500)
log_price.dropna(inplace=True)
tsplot(log_price)

#%%
def findpeaks(series, DELTA):
    """
    Finds extrema in a pandas series data.

    Parameters
    ----------
    series : `pandas.Series`
        The data series from which we need to find extrema.

    DELTA : `float`
        The minimum difference between data values that defines a peak.

    Returns
    -------
    minpeaks, maxpeaks : `list`
        Lists consisting of pos, val pairs for both local minima points and
        local maxima points.
    """
    # Set inital values
    mn, mx = np.Inf, -np.Inf
    minpeaks = []
    maxpeaks = []
    lookformax = True
    start = True
    # Iterate over items in series
    for time_pos, value in series.iteritems():
        if value > mx:
            mx = value
            mxpos = time_pos
        if value < mn:
            mn = value
            mnpos = time_pos
        if lookformax:
            if value < mx-DELTA:
                # a local maxima
                maxpeaks.append((mxpos, mx))
                mn = value
                mnpos = time_pos
                lookformax = False
            elif start:
                # a local minima at beginning
                minpeaks.append((mnpos, mn))
                mx = value
                mxpos = time_pos
                start = False
        else:
            if value > mn+DELTA:
                # a local minima
                minpeaks.append((mnpos, mn))
                mx = value
                mxpos = time_pos
                lookformax = True
    # check for extrema at end
    if value > mn+DELTA:
        maxpeaks.append((mxpos, mx))
    elif value < mx-DELTA:
        minpeaks.append((mnpos, mn))
    return minpeaks, maxpeaks

# findpeaks(log_price, 4)
# series = log_price
series = df_data.returns
minpeaks, maxpeaks = findpeaks(series, DELTA=0.5)
# Plotting the figure and extremum points
fig, ax = plt.subplots()
ax.set_ylabel('SP500 Log Price')
ax.set_xlabel('Time')
ax.set_title('Peaks in TimeSeries')
series.plot()
ax.scatter(*zip(*minpeaks), color='red', label='min')
ax.scatter(*zip(*maxpeaks), color='green', label='max')
ax.legend()
ax.grid(True)

plt.show()

# %%
# break down stationarity testing


def break_down_stationarity_test(ts, n):

    time_series = ts.copy()
    # example: we want 50 samples
    window_size = n
    # -- window walking
    q, r = divmod(len(ts), window_size)
    intervals = np.arange(0, len(ts), q)
    # intervals[-1] = intervals[-1] + r
    list_windows = []
    # print(intervals)
    list_adft_results = []
    for i in np.arange(1, len(intervals)):

        if i < len(intervals) - 1:
            temp = time_series.iloc[intervals[i-1]:intervals[i]]
        else:
            temp = time_series.iloc[intervals[i-1]:(intervals[i] + r)]
        list_windows.append(temp)

        regressions = ["nc", "c", "ct"]
        for index, value in enumerate(regressions):
            # print('Results Dickey Fuller Test - ' + value)
            regression_type = value
            df_test_DF = adfuller(time_series[intervals[i-1]:intervals[i]], autolag='AIC', regression=regression_type)
            df_output_results = pd.Series(df_test_DF[0:4], index=['Test Statistics', 'p-value', '#Lags Used', 'Number of Obs Used'])


            for key, value in df_test_DF[4].items():

                df_output_results['Criticial Value (%s)' % key] = value

            df_output_results['Test_Model'] = value
            df_output_results['from_index'] = intervals[i - 1]
            df_output_results['to_index'] = intervals[i]
            list_adft_results.append(df_output_results)

    return pd.DataFrame(list_adft_results)

log_value = np.log(df_data.SP500)
# log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
log_value.dropna(inplace=True)
df_all_results = break_down_stationarity_test(log_value, 20)
# df_all_results['Stationary'] =


#%%