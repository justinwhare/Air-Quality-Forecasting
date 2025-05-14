import pandas as pd

def get_pollutant_data(pollutant, lag, window_size, exog = False):
    '''
    pollutant:      (str)
    lag:            (int) number of previous timesteps to include, in hours
    window_size:    (int) size of window for rolling variable calculation, in hours
    exog:           (bool) include exogenous features
    '''

    #exogenous features can only be included if they would be known at time of forecast,
    #a multistep forecasting model wouldn't have access to exogenous data from the future.
    #the exogenous features will be included for the single time step model.
    if exog == True:
        cols = [pollutant, 'datetime', 'station', 'year', 'month', 'day', 'hour', 'TEMP', 'DEWP', 'PRES', 'WSPM', 'RAIN', 'wd']
    elif exog == False:
        cols = [pollutant, 'datetime', 'station', 'year', 'month', 'day', 'hour']
    

    data = pd.read_csv('output_data/cleaned_interp.csv', usecols = cols, parse_dates = ['datetime'])
   

    #creating time based, lag, and rolling window features to be used in the autoregressive model
    data['day_of_week'] = data['datetime'].dt.dayofweek + 1

    data.sort_values(['station', 'datetime'], inplace = True)
    for i in range(1, lag + 1):
        data[f'lag{i}'] = data.groupby('station')[pollutant].shift(i)

    rolling_mean = data.groupby('station')[pollutant].rolling(window = window_size).mean()
    data[f'roll_mean_{window_size}'] = rolling_mean.droplevel(0)


    #creating lag and rolling window features for exogenous features if included
    if exog == True:
        data.sort_values(['station', 'datetime'], inplace = True)
        for i in range(1, lag + 1):
            data[f'TEMP_lag{i}'] = data.groupby('station')['TEMP'].shift(i)

        rolling_mean = data.groupby('station')['TEMP'].rolling(window = window_size).mean()
        data[f'TEMP_roll_mean_{window_size}'] = rolling_mean.droplevel(0)


    #creating target variable as pollutant level from future time step
    data['target'] = data.groupby('station')[pollutant].shift(-1)


    #dropping the NaN values created by the lag feature creation
    data.dropna(inplace = True)


    #dropping the station column only from the univariate dataset and aggregating the 12 station timeseries into a single timeseries 
    if exog == False:
        data = data.drop(['station', 'year', 'month', 'day', 'hour', 'day_of_week'], axis = 1).groupby('datetime').mean()
        

    #cast object columns to categorical data type for handling by xgboost
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].astype('category')

    return data
    
def split_data(data, train_end, val_end):
    '''
    data:       (df) dataframe to be split
    train_end:  (str) right boundary for training set
    val_end:    (str) right boundary for validation set
    '''
    
    temp_df = data#.set_index(pd.to_datetime(data['datetime']))
    #temp_df.sort_index(inplace = True)
    #temp_df.drop('datetime', axis = 1, inplace = True)
    
    data_train = temp_df.loc[:train_end, :]
    data_val = temp_df.loc[train_end:val_end, :]
    data_test = temp_df.loc[val_end:, :]

    print(f"Dates train      : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
    print(f"Dates validation : {data_val.index.min()} --- {data_val.index.max()}  (n={len(data_val)})")
    print(f"Dates test       : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

    return data_train, data_val, data_test

