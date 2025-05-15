import pandas as pd

def get_pollutant_data(pollutant, lag = None, window_size = None, exog = False):
    '''
    Function to gather raw pollutant data into a single dataframe

    Args: 
        pollutant:      (str) Name of pollutant to be included in dataframe, 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'
        lag:            (int) number of previous timesteps to include as lagged variables, in hours
        window_size:    (int) size of window for rolling variable calculation, in hours
        exog:           (bool) include exogenous features, 'TEMP', 'DEWP', 'PRES', 'WSPM', 'RAIN', 'year', 'month', 'day', 'hour'

    Returns:
        data:           (pd.Dataframe) Dataframe containing pollutant data aggregated over 12 stations and including possible time based and exogenous features
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
        if lag:
            for i in range(1, lag + 1):
                df[f'TEMP_lag{i}'] = df['TEMP'].shift(i)
        
        if window_size:
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
    Splits dataframe into train, validation, and test splits

    Args:
        data:       (pd.Dataframe) dataframe to be split
        train_end:  (str) right boundary for training set
        val_end:    (str) right boundary for validation set
    
    Returns:
        data_train: (pd.Dataframe) Training Set
        data_val:   (pd.Dataframe) Validation Set
        data_test:  (pd.Dataframe) Test Set
    '''
    
    data_train = data.loc[:train_end, :]
    data_val = data.loc[train_end:val_end, :]
    data_test = data.loc[val_end:, :]

    print(f"Training Range      : {data_train.index.min()} to {data_train.index.max()}  (n={len(data_train)})")
    print(f"Validation Range    : {data_val.index.min()} to {data_val.index.max()}  (n={len(data_val)})")
    print(f"Test Range          : {data_test.index.min()} to {data_test.index.max()}  (n={len(data_test)})")

    return data_train, data_val, data_test