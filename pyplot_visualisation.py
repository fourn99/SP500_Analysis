import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.subplots as sb
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from preprocessing_data import *

#%%
df_data = df_time_series.copy()

# %%


# output grid of plot segments of data set
def subplot_data(m1, m2, data, x_axis_name, y_axis_name, title):
    plt.clf()
    rows = m1
    cols = m2
    grid_size = m1*m2
    size = len(data)
    quotient, remainder = divmod(size, grid_size)
    intervals = np.arange(0, size, quotient)

    fig = plt.figure()
    # fig.subplots_adjust()
    fig.subplots_adjust(left=0.125, bottom=0.5, right=0.9, top=0.9, wspace=0.9, hspace=0.9)
    mid_row = m1 / 2 + 1
    mid_col = m2 / 2 +1

    for k in range(1, grid_size+1):
        ax = fig.add_subplot(cols, rows, k)
        ax.plot(data[intervals[k-1]:intervals[k]])
        plt.setp(ax.get_xticklabels(), rotation=45)

    fig.text(x=0.5, y=0.0008, horizontalalignment='center', verticalalignment='bottom', s=x_axis_name)
    fig.suptitle(title)
    # fig.tight_layout(pad=20.0)
    plt.show()

    # ax.set(xlabel="Date",
    #        ylabel="Precipitation (inches)",
    #        title="Daily Total Precipitation\nBoulder, Colorado in July 2018")

subplot_data(3, 3, df_data['SP500'], 'Months', 'Price', 'Price Time Series')
subplot_data(3, 3, df_data['returns'], 'Months', 'Returns', 'Returns Time Series')
subplot_data(3, 3, df_data['cum_returns'], 'Months', 'Cumulative Returns', 'CUMSUM Time Series')


# %%
# ---------------------------------- Plotty Graphs

# graph with range slider
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_data.index, y=df_data.returns, name="Returns SP500", line_color='deepskyblue'))
fig.add_trace(go.Scatter(x=df_data.index, y=df_data.returns.rolling(window=12).mean(), name="Rolling Mean", line_color='red'))
fig.add_trace(go.Scatter(x=df_data.index, y=df_data.returns.rolling(window=12).std(), name="Rolling Std. Dev", line_color='green'))
# log_returns = np.log(df_data.returns)
# log_returns.dropna(inplace=True)
# fig.add_trace(go.Scatter(x=df_data.index, y=log_returns, name='Log Returns',
#                          line_color='dimgray'))

fig.update_layout(title_text='SP500 Time Series Analysis', xaxis_rangeslider_visible=True)
fig.show()
folder_name = 'C:\\Users\\nic_f\\PycharmProjects\\sp_timeseries_analysis\\data\\single_ts_slider' + '.html'
plotly.offline.plot(fig, filename=folder_name)
#%%
phase_1 = df_data.returns[0:250]
mean_1 = phase_1.mean()
print(mean_1)

phase_2 = df_data.returns[250:500]
mean_2 = phase_2.mean()
print(mean_2)

df_data.returns.rolling(window=100).mean().plot()
plt.show()

#%%

def subplot_data_plotly(m1, m2, data, x_axis_name, y_axis_name, title):
    rows = m1
    cols = m2
    grid_size = m1*m2
    size = len(data)
    quotient, remainder = divmod(size, grid_size)
    intervals = np.arange(0, size, quotient)
    print(intervals)

    get_n_biggest_drawdown(data.cumsum(), 1)

    fig_01 = sb.make_subplots(rows, cols, print_grid=True)
    k = 1

    for r1 in range(1, rows+1):
        for c1 in range(1, cols+1):
            trace = go.Scatter(x=data[intervals[k-1]:intervals[k]].index, y=data[intervals[k-1]:intervals[k]].values)

            temp_cumsum = data[intervals[k-1]:intervals[k]]
            print(temp_cumsum)
            df_temp = get_n_biggest_drawdown(temp_cumsum, 1)
            fig_01.append_trace(trace, r1, c1)

            for row, col in df_temp.iterrows():
                fig_01.append_trace(go.Scatter(x=[col[0], col[1]], y=[temp_cumsum[col[0]], temp_cumsum[col[1]]],
                                               mode='markers', name=('DD' + str(row))), r1, c1)
            k = k + 1

    fig_01.update_layout(title_text='SP500 Time Series Analysis', height=1000)

    #   Create Html
    folder_name = 'C:\\Users\\nic_f\\PycharmProjects\\sp_timeseries_analysis\\data\\single_ts_breakdown' + '.html'
    plotly.offline.plot(fig_01, filename=folder_name)


log_price = np.log(df_data.SP500)
log_price = log_price.replace([np.inf, -np.inf], np.nan)
log_price.dropna(inplace=True)
subplot_data_plotly(2,1, log_price, 'Months', 'Price', 'Log Price Time Series')
#   Layout setting
# fig.update_layout(title_text='SP500 Time Series Analysis', height=1000)

# fig.update_yaxes(type="log", row=1, col=2)
#%%
from arch import arch_model
r=df_data.returns * 100
garch11 = arch_model(r, p=1, q=1)
res = garch11.fit(update_freq=10)
print(res.summary())


#%%
