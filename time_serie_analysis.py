import pandas as pd

class TimeSeriesAnalysis:
    
    time_series = None
    
    def __init__(self, df_data):
        # get data from excel file
        df_data_raw = df_data.copy()
        df_data_raw.rename(columns={'Actual Date': 'Date'}, inplace=True)

        # cast string to datetime object
        df_data_raw.Date = pd.to_datetime(df_data_raw.Date, infer_datetime_format='YYYY-MM-DD')
        time_series = df_data_raw.copy()

        time_series.set_index('Date', inplace=True)

        # calculate returns
        time_series['returns'] = time_series['SP500'].pct_change()

        # drop rows with nan value in any columns
        time_series.dropna(axis='rows', inplace=True)

        # compute absolute returns
        time_series['abs_returns'] = time_series['returns'].abs()

        # calculate difference with lag 1
        time_series['delta'] = time_series['SP500'].diff(1)

        # compute cumulative returns
        time_series['cum_returns'] = time_series['returns'].cumsum()

        print('Final DataFrame')
        print(time_series.describe())

    # # compute log price, drops inf and nan values, returns series
    # def get_log_price(self):
    #
    # def get_log_returns(self):
    #
    # # returns dates and magnitude of drawdowns
    # def compute_drawdowns(self):
    #
    # #  create a subplot of nxm in plotly
    # def subplot_series_plotly(self):
    #
    # def compute_hurst_coefficient(self):
    #
    # def stationarity_adf_test(self):
    #
    # def test_power_law(self):
