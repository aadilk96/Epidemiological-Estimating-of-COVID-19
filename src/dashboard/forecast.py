import pandas as pd
import numpy as np

def create_dataset_country(data, window):
    """ Create a dataset using a sliding window over the given data.
    
    Params:
        data (numpy: T): array with case number time series.
        window (int): size of the sliding window, i.e. lookback for predictions.
    Returns:
        x (numpy: T' x W): array with T' samples (T'=T-W), each with W values (window size)
    """
    dataX, dataY = [], []
    for i in range(len(data)-window):
        a = data[i:(i+window)]
        dataX.append(a)
        dataY.append(data[i + window])
    return np.array(dataX), np.array(dataY)

def fit_country_krr(window):
    csv_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    confirmed = pd.read_csv(csv_url)
    confirmed.index = confirmed['Country/Region']
    confirmed = confirmed.drop(['Province/State', 'Lat', 'Long', 'Country/Region'], axis=1)
    confirmed = confirmed.T
    data = confirmed.to_numpy()

    window_size = window
    xy_per_country = [create_dataset_country(country_col, window_size) for country_col in data.T]
    N_TOTAL = xy_per_country[0][0].shape[0]
    N_TRAIN = int(N_TOTAL*.9)

    trainx_countries = np.concatenate([x[:N_TRAIN] for x,y in xy_per_country], axis=0)
    trainy_countries = np.concatenate([y[:N_TRAIN] for x,y in xy_per_country], axis=0)
    testx_countries = np.concatenate([x[N_TRAIN:] for x,y in xy_per_country], axis=0)
    testy_countries = np.concatenate([y[N_TRAIN:] for x,y in xy_per_country], axis=0)

    global_confirmed = confirmed.sum(axis=1)
    global_data = global_confirmed.to_numpy()
    x_global, y_global = create_dataset_country(global_data, window_size)

    trainx_global = x_global[:N_TRAIN]
    trainy_global = y_global[:N_TRAIN]
    testx_global = x_global[N_TRAIN:]
    testy_global = y_global[N_TRAIN:]

    GLOBAL_IMPORTANCE = 1
    np.concatenate([np.repeat(trainx_global, 10, axis=0), trainx_global], axis=0).shape

    trainx = np.concatenate([
        np.repeat(trainx_global, GLOBAL_IMPORTANCE, axis=0), trainx_countries
    ], axis=0)
    trainy = np.concatenate([
        np.repeat(trainy_global, GLOBAL_IMPORTANCE, axis=0), trainy_countries
    ], axis=0)

    testx = np.concatenate([
        np.repeat(testx_global, GLOBAL_IMPORTANCE, axis=0), testx_countries
    ], axis=0)
    testy = np.concatenate([
        np.repeat(testy_global, GLOBAL_IMPORTANCE, axis=0), testy_countries
    ], axis=0)

    krr = KernelRidge()
    krr.fit(trainx, trainy)
    country_pred = confirmed.copy()
    country_pred = confirmed.groupby('Country/Region', axis=1).sum()
    country_pred = country_pred.tail(window_size)
    country_forecast_dic = {}
    
    for name in country_pred.columns:
        pred = krr.predict([country_pred[name].values])
        country_forecast_dic[name] = pred.item()
        
    return country_forecast_dic

def getCountryPredVals(window_size):
    csv_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    confirmed = pd.read_csv(csv_url)
    confirmed.index = confirmed['Country/Region']
    confirmed = confirmed.drop(['Province/State', 'Lat', 'Long', 'Country/Region'], axis=1)
    confirmed = confirmed.T  
    country_pred = confirmed.copy()
    country_pred = confirmed.groupby('Country/Region', axis=1).sum()
    country_pred = country_pred.tail(window_size)
    return country_pred
    