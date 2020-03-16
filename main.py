#%%
# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import plotly
import plotly.subplots as sb
import plotly.graph_objects as go
import plotly.express as px
from sklearn import metrics
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# import data from other scripts
from time_serie_analysis import *
from preprocessing_data import *


#%%
# -- get processed data from script
df_data = df_time_series.copy()

freq,bins,_ = plt.hist(df_data.returns, bins=50)
plt.show()

df_histo = pd.DataFrame({'Frequency':freq}, index=bins[0:50])

median = np.median(df_data.returns)
upper_quartile = np.percentile(df_data.returns, 75)
lower_quartile = np.percentile(df_data.returns, 25)

iqr = upper_quartile - lower_quartile
upper_whisker = df_data.returns[df_data.returns<=upper_quartile+1.5*iqr].max()
lower_whisker = df_data.returns[df_data.returns>=lower_quartile-1.5*iqr].min()

nb_neg_outliers = df_histo.loc[df_histo.index < lower_whisker].sum()
nb_pos_outliers = df_histo.loc[df_histo.index > upper_whisker].sum()
nb_neg_whisker = df_histo.loc[(df_histo.index > lower_whisker) & (df_histo.index < lower_quartile)].sum()
nb_pos_whisker = df_histo.loc[(df_histo.index < upper_whisker) & (df_histo.index > upper_quartile)].sum()
nb_middle = df_histo.loc[(df_histo.index < upper_quartile) & (df_histo.index > lower_quartile)].sum()
list_pie_chart = [nb_neg_outliers[0], nb_pos_outliers[0], nb_neg_whisker[0], nb_pos_whisker[0], nb_middle[0]]
# box = plt.pie(list_pie_chart, labels=['(-) outliers', '(+) Outliers', '(-) whisker', '(+) whisker', 'middle'])

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)

labels=['(-) outliers', '(+) Outliers', '(-) whisker', '(+) whisker', 'middle']
wedges, texts, autotexts = plt.pie(list_pie_chart,
                                   autopct=lambda pct: func(pct, list_pie_chart), textprops=dict(color="w"))
plt.legend(wedges, labels,
          title="Category",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(autotexts, size=12, weight="bold")
plt.show()
# nb_0_0_5 =
# df_histo.plot.pie(y='Frequency', labels=df_histo.index)
# plt.show()
#%%
# -- Plotly dashboard of time series to analyse
# n_lag = len(df_data['SP500'])
# acf_data = acf(df_data['returns'], missing='drop', nlags=n_lag, fft=True)  # compute acf data

# Add graph data
trace_1 = go.Scatter(x=df_data.index, y=df_data.SP500, name="Price SP500", line_color='deepskyblue')

trace_2 = go.Scatter(x=df_data.index, y=df_data.returns, name='Returns', line_color='dimgray')

trace_3 = go.Scatter(x=df_data.index, y=df_data.abs_returns, name='|Returns|', line_color='red')

trace_4 = go.Scatter(x=df_data.index, y=df_data.cum_returns, name='Cumulative Returns', line_color='orange')

temp_log_returns = np.log(df_data.returns)
temp_log_returns.dropna(inplace=True)
trace_5 = go.Scatter(x=df_data.index, y=temp_log_returns, name='Log Returns', line_color='green')

# trace_6 = go.Bar(x=np.arange(1, n_lag), y=acf_data, name='ACF')
trace_6 = go.Box(x=np.array(df_data.returns), name='Box Plot', boxpoints='all')

trace_7 = go.Histogram(x=df_data.returns)

trace_8 = go.Scatter(x=df_data.index, y=df_data.SP500, name="Price SP500", line_color='deepskyblue')

# Subplots
fig = sb.make_subplots(
    rows=5,
    cols=2,
    print_grid=True,
    subplot_titles=('Prices', 'Prices - Log Scale', ' Returns ',  'Absolute Returns', 'Cumulative Returns',
                    'Distribution Returns', 'Log Returns', 'Box Plot Returns')
)

# for loop to add drawdown to dashboard
# for row, col in df_dd.iterrows():
#     fig.append_trace(go.Scatter(x=[col[0], col[1]], y=[df_data.cum_returns[col[0]], df_data.cum_returns[col[1]]],
#                                 mode='markers', name=('DD' + str(row))), 3, 1)


fig.append_trace(trace_8, 1, 1)
fig.append_trace(trace_1, 1, 2)
fig.append_trace(trace_2, 2, 1)
fig.append_trace(trace_3, 2, 2)
fig.append_trace(trace_4, 3, 1)
fig.append_trace(trace_7, 3, 2)
fig.append_trace(trace_5, 4, 1)
fig.append_trace(trace_6, 4, 2)

#   Layout setting
fig.update_layout(title_text='SP500 Time Series Analysis', height=1000)

fig.update_yaxes(type="log", row=1, col=2)

#   Create Html
folder_name = 'C:\\Users\\nic_f\\PycharmProjects\\sp_timeseries_analysis\\data\\dashboards' + '.html'
plotly.offline.plot(fig, filename=folder_name)


#%%
# -- Distribution Testing
# Power Law Distribution Testing

# cast returns into numpy array
arr_returns = df_data.returns.to_numpy()

# fit returns to power law
fit = powerlaw.Fit(arr_returns)

alpha = fit.alpha  # exponent
sigma = fit.sigma  # std. error

fig2 = fit.plot_pdf(color='b', linewidth=2)
fit.power_law.plot_pdf(color='r', linestyle='--', ax=fig2)
plt.title('PDF - Power Law Fit')
plt.show()

R, p = fit.distribution_compare('power_law', 'exponential')#, normalized_ratio=True)

print('Alpha: ' + str(alpha) + " Sigma: " + str(sigma) + " R: " + str(R) + " P-value: " + str(p))

fit.distribution_compare('power_law', 'exponential')
fig4 = fit.plot_ccdf(linewidth=3, label='PL fit Expo')
fit.power_law.plot_ccdf(ax=fig4, color='r', linestyle='--', label='Power Law CCDF')
fit.exponential.plot_ccdf(ax=fig4, color='g', linestyle=':', label='Exponential CCDF')
plt.title('Power Law vs Exponential Distribution')
plt.legend(loc='best')
plt.show()

#%%
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

    # plt.clf()
    # plot_acf(time_serie)
    # plt.title('ACF')
    # plt.show()

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
        plt.title(name_series +' ADF Test ' + regression_type + ' - P-Value: ' + str(round(df_output_results[1],5)))
        for index, value in enumerate(height):
            plt.text(value, index, str(round(value,3)))
        print(df_output_results)
        print('\n')
        plt.show()


#%%
ts_stationarity_test(df_data.SP500, "Price")

'''
Results: non-stationary all around
'''

#%%
log_price = np.log(df_data.SP500)
log_price.dropna(inplace=True)
ts_stationarity_test(log_price, "Log Price")  # ADF on Log Price

'''
Results indicate non-stationary for No constant, no trend; constants, no trend
Stationary with constant and with trend
'''


#%%

df_results_adf = ts_stationarity_test(df_data.returns, "Returns")  # ADF on Returns

'''
Results indicates stationary for all regression types
'''

#%%
log_returns = np.log(df_data.returns)
log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
log_returns.dropna(inplace=True)

ts_stationarity_test(log_returns, "Log Returns")  # ADF on Log Returns

'''
Results: non-stationary no constant/no trend
stationary for constant, no trend; constant and trend;
'''

#%%

# -- Hurst Coefficient



#%%
# --------- Auto Correlation Functions

plot_acf(df_data.returns)
plt.title('ACF - Returns')
plt.show()

plot_acf(df_data.delta)
plt.title('ACF - Delta')
plt.show()

#%%
# ----------------------- DRAW DOWN IDENTIFICATION
# def get_drawdown(ts, window_size):
#     # -- window walking
#     # q, r = divmod(len(ts), window_size)
#     # intervals = np.arange(0, len(ts), window_size)
#     # list_windows = []
#     # print(intervals)
#     #
#     # # print(ts.iloc[intervals[0]:intervals[1]])
#     # # print()
#     #
#     # for i in np.arange(1, len(intervals)):
#     #
#     #     if i < len(intervals) - 1:
#     #         temp = ts.iloc[intervals[i-1]:intervals[i]]
#     #     else:
#     #         temp = ts.iloc[intervals[i-1]:(intervals[i] + r)]
#     #
#     #     list_windows.append(temp)
#     #     # print(temp)

# n = 1000
# xs = np.random.randn(n).cumsum()
# i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
# j = np.argmax(xs[:i]) # start of period
#
# plt.plot(xs)
# plt.plot([i, j], [xs[i], xs[j]], 'o', color='Red', markersize=10)
# plt.show()


def get_n_biggest_drawdown(cum_ts, n_biggest):

    temp_series = cum_ts.copy()
    list_beg_date = []
    list_end_date = []
    list_beg_value = []
    list_end_value = []

    for i in np.arange(n_biggest):

        end_date = np.argmax(np.maximum.accumulate(temp_series) - temp_series)
        beg_date = np.argmax(temp_series[:end_date])

        list_beg_date.append(beg_date)
        list_end_date.append(end_date)
        list_beg_value.append(temp_series[beg_date])
        list_end_value.append(temp_series[end_date])

        print('Biggest DD ' + str(i) + ' From ' + str(beg_date) + ' to ' + str(end_date))

        temp_series.drop(temp_series[beg_date:end_date].index, inplace=True)
        print(len(temp_series))

    return pd.DataFrame({'start': list_beg_date, 'end': list_end_date, 'peak': list_beg_value, 'bottom': list_end_value})

# log_price = np.log(df_data.SP500)
# log_price.dropna(inplace=True)


df_dd = get_n_biggest_drawdown(cum_ts=df_data.cum_returns, n_biggest=5)
plt.plot(df_data.cum_returns)

for row, col in df_dd.iterrows():
    plt.plot([col[0], col[1]], [df_data.cum_returns[col[0]], df_data.cum_returns[col[1]]], 'o', color='Red', markersize=10)

plt.show()

#%%
def compute_drawdowns(time_series_dd, drop_by, drop_to, peak_since):

    temp_ts = time_series_dd.copy()


'''
drop_by = a drop in price of 25%, i.e. down to 0.75 of the peak price, which is in line
with the 1987 crash,
drop_to =  a period of 60 weekdays within which the drop in price needs to occur.
peak_since = a period of 262 weekdays prior to the peak for which there is no value higher
than the peak,
'''
compute_drawdowns(df_data.SP500, 0.75, 100, 200)

plt.plot(df_data.returns.cumsum())
plt.show()
ts_stationarity_test(df_data.returns.cumsum(), "Cum Returns")
#%%
# ------------------- Chow test
from sklearn.linear_model import LinearRegression as lr


def linear_residuals(X, y):

    # fits the linear model
    model = lr().fit(X, y)
    # creates a dataframe with the predicted y in a column called y_hat
    summary_result = pd.DataFrame(columns=['y_hat'])
    yhat_list = [float(i[0]) for i in np.ndarray.tolist(model.predict(X))]
    summary_result['y_hat'] = yhat_list
    # saves the actual y values in the y_actual column
    summary_result['y_actual'] = y.values
    # calculates the residuals
    summary_result['residuals'] = summary_result.y_actual - summary_result.y_hat
    # squares the residuals
    summary_result['residuals_sq'] = summary_result.residuals ** 2

    return (summary_result)


def calculate_RSS(X, y):
    # calls the linear_residual function
    resid_data = linear_residuals(X, y)
    # calculates the sum of squared resiudals
    rss = resid_data.residuals_sq.sum()

    # returns the sum of squared residuals
    return (rss)


# defines a function to return the p-value from a Chow Test
def ChowTest(X, y, last_index_in_model_1, first_index_in_model_2):
    # gets the RSS for the entire period
    rss_pooled = calculate_RSS(X, y)

    # splits the X and y dataframes and gets the rows from the first row in the dataframe
    # to the last row in the model 1 testing period and then calculates the RSS
    X1 = X.loc[:last_index_in_model_1]
    y1 = y.loc[:last_index_in_model_1]
    rss1 = calculate_RSS(X1, y1)

    # splits the X and y dataframes and gets the rows from the first row in the model 2
    # testing period to the last row in the dataframe and then calculates the RSS
    X2 = X.loc[first_index_in_model_2:]
    y2 = y.loc[first_index_in_model_2:]
    rss2 = calculate_RSS(X2, y2)

    # gets the number of independent variables, plus 1 for the constant in the regression
    k = X.shape[1] + 1
    # gets the number of observations in the first period
    N1 = X1.shape[0]
    # gets the number of observations in the second period
    N2 = X2.shape[0]

    # calculates the numerator of the Chow Statistic
    numerator = (rss_pooled - (rss1 + rss2)) / k
    # calculates the denominator of the Chow Statistic
    denominator = (rss1 + rss2) / (N1 + N2 - 2 * k)

    # calculates the Chow Statistic
    Chow_Stat = numerator / denominator

    # Chow statistics are distributed in a F-distribution with k and N1 + N2 - 2k degrees of
    # freedom
    from scipy.stats import f

    # calculates the p-value by subtracting 1 by the cumulative probability at the Chow
    # statistic from an F-distribution with k and N1 + N2 - 2k degrees of freedom
    p_value = 1 - f.cdf(Chow_Stat, dfn=5, dfd=(N1 + N2 - 2 * k))

    # saves the Chow_State and p_value in a tuple
    result = (Chow_Stat, p_value)

    # returns the p-value
    return (result)

results_chow_test = ChowTest(X=df_data.returns.index, y=df_data.returns.values, last_index_in_model_1=250, first_index_in_model_2=300)


#%%
# lin regression prediction
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

log_price = np.log(df_data.SP500)
log_price.dropna(inplace=True)
ys = np.array(log_price)
xs = np.linspace(0, 1, len(log_price))

ys = ys.reshape(-1, 1)
xs = xs.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=0)
reg = LinearRegression()
reg.fit(X_train, y_train)
print('Intercept: ' + str(reg.intercept_))
print('Slope: ' + str(reg.coef_))
y_pred = reg.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

#  following works
# ys = np.array(log_price)
# xs = np.linspace(0, 1, len(log_price))
# slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
# print('Slope: '+str(slope) + ' Intercept: ' + str(intercept) + ' R_Value: ' + str(r_value) + ' P-Value: ' + str(p_value) + ' STD Error: ' + str(std_err) )
#
# plt.clf()
# y = slope*xs+intercept
# plt.plot(xs, y, label='Liner Regression')
# plt.plot(xs, ys, label='Log Price')
# plt.legend(loc='best')
# plt.show()
#%%

#%%
list_index = []
list_bits = []

for index, value in df_data.returns.iteritems():
    # print("index: " + str(index) + " Value: " + str(value))
    list_index.append(index)
    if value < 0:
        list_bits.append(-1)
    elif value == 0:
        list_bits.append(0)
    else:
        list_bits.append(1)
df_bits = pd.DataFrame(data=list_bits, index=list_index)

plt.plot(df_bits.cumsum())
plt.show()
#%%
# -- Stationary Test DF
#todo to fix LATER
def ts_stationarity_test(time_serie, name_series, lags=None, figsize=(10, 8), style='bmh'):
    moving_avg = time_serie.rolling(window=12).mean()
    moving_std = time_serie.rolling(window=12).std()
    # -- Augmented DF test
    # c = constant, ct = constant + trend , ctt =  constant, and linear and quadratic trend., nc = no constant}
    regressions = ["nc", "c", "ct", "ctt"]
    for index, value in enumerate(regressions):
        #         print('Results Dickey Fuller Test - ' + value)
        regression_type = value
        df_test_DF = adfuller(time_serie, autolag='AIC', regression=regression_type)
        df_output_results = pd.Series(df_test_DF[0:4],
                                      index=['Test Statistics', 'p-value', '#Lags Used', 'Number of Obs Used'])

        for key, value in df_test_DF[4].items():
            df_output_results['Criticial Value (%s)' % key] = value

        # data for bar plot of results
        x = [df_output_results.index[0], df_output_results.index[4], df_output_results.index[5],
             df_output_results.index[6]]
        height = [df_output_results[0], df_output_results[4], df_output_results[5], df_output_results[6]]

        plt.barh(x, height)
        plt.title(name_series + ' ADF Test ' + regression_type + ' - P-Value: ' + str(round(df_output_results[1], 5)))
        for index, value in enumerate(height):
            plt.text(value, index, str(round(value, 3)))

        plt.show()
        print(df_output_results)
        print('\n')

    if not isinstance(time_serie, pd.Series):
        time_serie = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        adf_nc = plt.subplot2grid(layout, (1, 0))
        adf_c = plt.subplot2grid(layout, (1, 1))
        adf_ct = plt.subplot2grid(layout, (2, 0))
        #         pp_ax = plt.subplot2grid(layout, (2, 1))

        time_serie.plot(ax=ts_ax)
        ts_ax.plot(moving_avg, color='red', label='Rolling Mean 12M')
        ts_ax.plot(moving_std, color='black', label='Rolling Stand. Dev 12M')
        ts_ax.set_title('Time Series Analysis Plots')
        ts_ax.legend(loc='best')
        #         smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        #         smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        #         sm.qqplot(y, line='s', ax=qq_ax)
        #         qq_ax.set_title('QQ Plot')
        #         scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
        plt.show()


log_price = np.log(df_data.SP500)
log_price.dropna(inplace=True)
df_results_adf = ts_stationarity_test(log_price, 'log_price')

#%%