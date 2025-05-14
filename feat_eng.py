import pandas as pd

def get_pollutant_data(pollutant, lag = None, window_size = None, exog = False):
    '''
    pollutant:      (str)
    lag:            (int) number of previous timesteps to include, in hours
    window_size:    (int) size of window for rolling variable calculation, in hours
    exog:           (bool) include exogenous features
    '''

    data = pd.read_csv('output_data/cleaned_interp.csv', parse_dates = ['datetime'])
    pollutant_list = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

    #aggregate the dataset over the 12 stations to work with a single mean time series for simplicity
    df = data[list(data.select_dtypes(include='float64').columns) + ['datetime']]
    df = df.groupby('datetime').mean()

    #creating time based lag and rolling window features 
    if lag:
        for i in range(1, lag + 1):
            df[f'lag_{i}'] = df[pollutant].shift(i)
    
    if window_size:
        df[f'roll_mean_{window_size}'] = df[pollutant].rolling(window = window_size).mean()

    #engineering optional exogenous features
    if exog == True:
        for i in range(1, lag + 1):
            df[f'TEMP_lag{i}'] = df['TEMP'].shift(i)
        
        df[f'TEMP_roll_mean'] = df['TEMP'].rolling(window = window_size).mean()

        #re-adding the non numerical columns after the aggregation
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['hour'] = df.index.hour

        #casting columns with object dtype to category for handling by xgboost
        for col in df.select_dtypes(include = 'object').columns:
            df[col] = df[col].astype('category')

    #creating target variable as the next pollutant concentration value in the time series
    df['target'] = df[pollutant].shift(-1)

    #drop rows with NaN values formed by creation of lag and rolling features
    df.dropna(inplace = True)

    if exog == False:
        pollutant_list.remove(pollutant)
        return df.drop(['TEMP', 'DEWP', 'PRES', 'WSPM', 'RAIN'] + pollutant_list, axis = 1)
    elif exog == True:
        return df

    
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

#print(get_pollutant_data('PM2.5').index)