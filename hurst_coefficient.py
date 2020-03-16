#%%
# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from preprocessing_data import df_time_series

#%%
df_data = df_time_series.copy()

#%%
# ---------------------------- Hurst Coefficient Computation
def compute_hurst(series_ts, power):

    ts_price = series_ts
    list_subseries = []
    # split time series into power of 2 (i.e.: first subset length is 2473, 2473/2....
    for i in np.arange(0, power):

        n = len(ts_price)
        ts_temp = ts_price.copy()
        partition_size = 2 ** i
        ts_partitioned = np.array_split(ts_temp, partition_size)
        list_subseries.append(ts_partitioned)

    list_sublist_name = []
    list_sublist_rs = []
    list_sublist_size = []

    for j in np.arange(len(list_subseries)):

        list_sublist_name.append(len(list_subseries[j]))
        list_temp_rs = []
        for k in np.arange(len(list_subseries[j])):
            sub_list = list_subseries[j][k]

            if k == 0:
                list_sublist_size.append(len(sub_list))

            m_j = np.mean(sub_list)  # compute mean
            s_j = np.sqrt(np.var(sub_list))  # compute std dev
            ts_adjusted = sub_list - m_j  # compute adjusted series
            ts_run_total = np.cumsum(ts_adjusted)  # compute runnig sum of adjusted serie
            run_max = np.max(ts_run_total)
            run_min = np.min(ts_run_total)
            run_range = run_max - run_min
            r_s = run_range / s_j
            list_temp_rs.append(r_s)

        list_sublist_rs.append(np.mean(list_temp_rs))

    df_rs = pd.DataFrame({'Range': list_sublist_name, 'Size':list_sublist_size, 'AVG_RS': list_sublist_rs})

    df_rs['Log_Size'] = np.log10(df_rs.Size)
    df_rs['Log_AVG_RS'] = np.log10(df_rs.AVG_RS)

    return df_rs


# -- Compute Hurst with returns
# df_results = compute_hurst(series_ts=df_data.returns, power=6)
df_results = compute_hurst(series_ts=df_data.SP500, power=6)

df_results_h1 = df_results.iloc[3:]
df_results_h2 = df_results.iloc[0:2]

df_results_t1 = df_results.iloc[4:6]
df_results_t2 = df_results.iloc[2:4]
df_results_t3 = df_results.iloc[0:2]

list_hurst_results = [df_results, df_results_h1, df_results_h2, df_results_t1, df_results_t2, df_results_t3]
list_name_results = ['Full Series', 'First Half', 'Second Half', 'First Third', 'Second Third', 'Third Third']
list_dict_result = []
for i in np.arange(len(list_hurst_results)):
    temp = list_hurst_results[i]
    slope, intercept, r_value, p_value, std_err = linregress(temp['Log_Size'], temp['Log_AVG_RS'])
    dict_results = {'Slope': slope, 'Intercept': intercept, 'R-Value': r_value, 'P-Value': p_value, 'Std Error': std_err
                    , 'Fractal Dimension': (2-slope), 'Time Period': list_name_results[i]}
    list_dict_result.append(dict_results)

df_table_all_results = pd.DataFrame(list_dict_result)
print(df_table_all_results.head(6))

slope, intercept, r_value, p_value, std_err = linregress(df_results['Log_Size'], df_results['Log_AVG_RS'])

plt.plot(df_results['Log_Size'], df_results['Log_AVG_RS'])
plt.ylabel('Log R/S')
plt.xlabel('Log Size')
plt.title('R/S Analysis SP500 - Returns From 1812 to 2018')
plt.text(2, 1.6, 'y = ' + str(round(slope, 3)) + 'x +' + str(round(intercept, 3)))
plt.text(2, 1.55, 'R^2 = ' + str(round(r_value ** 2, 3)) + '   P-value = ' + str(round(p_value, 5)))
plt.text(2, 1.5, 'Fractal Dimension: ' + str(round(2-slope, 3)))
plt.show()

# ---- Compute hurst with Log returns

# temp_ts = df_data['returns'].replace(0, 1)
# temp_ts = temp_ts.apply(lambda x: np.log10(x))
# temp_ts.dropna(axis='rows', inplace=True)
# df_results_1 = compute_hurst(series_ts=temp_ts, power=6)
# 
# slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = linregress(df_results_1['Log_Size'], df_results_1['Log_AVG_RS'])
# plt.clf()
# plt.plot(df_results_1['Log_Size'], df_results_1['Log_AVG_RS'])
# plt.ylabel('Log R/S')
# plt.xlabel('Log Size')
# plt.title('R/S Analysis SP500 Log Returns - 1812 to 2018')
# plt.text(1.8, 1.6, 'y = ' + str(round(slope_1, 3)) + 'x +' + str(round(intercept_1, 3)))
# plt.text(1.8, 1.55, 'R^2 = ' + str(round(r_value_1 ** 2, 3)) + '   P-value = ' + str(round(p_value_1, 5)))
# plt.text(1.8, 1.5, 'Fractal Dimension: ' + str(round(2-slope_1, 3)))
# plt.show()


#%%