import pandas as pd

def process_time_serie(ts):

    # calculate returns
    ts['returns'] = ts['SP500'].pct_change()

    # drop rows with nan value in any columns
    ts.dropna(axis='rows', inplace=True)

    # compute absolute returns
    ts['abs_returns'] = ts['returns'].abs()

    # calculate difference with lag 1
    ts['delta'] = ts['SP500'].diff(1)

    # compute cumulative returns
    ts['cum_returns'] = ts['returns'].cumsum()

    print('DataFrame To Analyse')
    print(ts.describe())
    return ts

# get data from excel file
df_data_raw = pd.read_excel('D:\\Finance_Project\\Data_SP500\\Raw_Data.xlsx', usecols=['Actual Date', 'SP500'])
df_data_raw.rename(columns={'Actual Date': 'Date'}, inplace=True)
# cast string to datetime object
df_data_raw.Date = pd.to_datetime(df_data_raw.Date, infer_datetime_format='YYYY-MM-DD')
df_time_series = process_time_serie(df_data_raw)
#%%
# df_data_yahoo = pd.read_csv('./data/sp_yahoo.csv')
# # print(df_data_yahoo.columns)
# df_data_yahoo.Date = pd.to_datetime(df_data_yahoo.Date, infer_datetime_format='YYYY-MM-DD')
# df_data_yahoo.set_index('Date', inplace=True)
# print(df_data_yahoo.head())
#
# max_date = df_time_series.last_valid_index()
# min_date = df_data_yahoo.first_valid_index()
#
# df_temp_originial = df_time_series[min_date: max_date]
# df_temp_yahoo = df_data_yahoo[min_date:max_date]
#
# print(df_temp_originial.tail())
#
# print(df_temp_originial.SP500.equals(df_temp_yahoo['Adj Close']))

#%%