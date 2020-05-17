import pandas as pd
import datetime
import numpy as np
import math
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
import statsmodels.api as sm
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing

def readRenameSumTotal(url):
    df = pd.read_csv(url)
    df.rename(columns={"Province/State":"Prov", "Country/Region":"Country"}, inplace=True)
    cases_sum = df.sum()[-1]
    df.loc["Total"] = df.sum()
    df.iloc[-1,1] = "Worldwide"
    return df, cases_sum

def countryGraphData(name, df):
    country = df[df["Country"] == name].groupby("Country").sum()
    graph = country.iloc[0,2:len(df)]
    return graph

def seirDf(s, e, i, r):
    seir_df = pd.DataFrame(data=s, columns={"Susceptible"})
    seir_df["Exposed"] = e
    seir_df["Infected"] = i
    seir_df["Removed"] = r
    return seir_df

def getDatewiseOverall(dataframes):
    labels = ['Confirmed', 'Deaths', 'Recovered']
    column_names = ["Date", "Datetime", "Confirmed", "Deaths", "Recovered"]
    dfs = []
    for i in range(len(dataframes)):
        series = dataframes[i].drop(['Lat', 'Long', 'Country', 'Prov'], axis=1).iloc[-1, :].T
        df = pd.DataFrame({'Date':series.index, labels[i]:series.values})
        dfs.append(df)
    df = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs)
    df["Datetime"]=pd.to_datetime(df["Date"])
    return df.reindex(columns=column_names)

def seirExplicit(timeperiod, meetings, infection_prob, removed_pop, infected_inp, latent_period, infectivity_period):
    if meetings is None:
        meetings = 1
    if infected_inp is None:
        infected_inp = 1
    if removed_pop is None:
        removed_pop = 1
        
    end_time = 200.0
    num_steps = int(end_time / timeperiod)
    times = timeperiod * np.array(range(num_steps + 1))

    s = np.zeros(num_steps + 1)
    e = np.zeros(num_steps + 1)
    i = np.zeros(num_steps + 1)
    r = np.zeros(num_steps + 1)

    s[0] = 7.8e9 - removed_pop
    e[0] = 0.
    i[0] = float(infected_inp) 
    r[0] = removed_pop

    transmission_coeff = ((float(infection_prob) / 100) * meetings) / 7.8e9
    for step in range(num_steps):
        s2e = timeperiod * transmission_coeff * s[step] * i[step]
        e2i = timeperiod / latent_period * e[step]
        i2r = timeperiod / infectivity_period * i[step]
        s[step + 1] = s[step] - s2e
        e[step + 1] = e[step] + s2e - e2i
        i[step + 1] = i[step] + e2i - i2r
        r[step + 1] = r[step] + i2r
    
    return pd.Series(s), pd.Series(e), pd.Series(i), pd.Series(r), times

def seirImplicit(timeperiod, meetings, infection_prob, removed_pop, infected_inp, latent_period, infectivity_period):
    if meetings is None:
        meetings = 1
    if infected_inp is None:
        infected_inp = 1
    if removed_pop is None:
        removed_pop = 1

    end_time = 300.0
    num_steps = int(end_time / timeperiod)
    times = timeperiod * np.array(range(num_steps + 1))

    s = np.zeros(num_steps + 1)
    e = np.zeros(num_steps + 1)
    i = np.zeros(num_steps + 1)
    r = np.zeros(num_steps + 1)

    s[0] = 7.8e9 - removed_pop
    e[0] = 0. 
    i[0] = float(infected_inp)
    r[0] = removed_pop 

    transmission_coeff = ((float(infection_prob) / 100) * meetings) / 7.8e9

    for step in range(num_steps):
        s2e = timeperiod * transmission_coeff * s[step] * i[step]
        e2i = timeperiod / latent_period * e[step]
        i2r = timeperiod / infectivity_period * i[step]

        p = ((1.0 + timeperiod / infectivity_period) / (timeperiod * transmission_coeff) + i[step]) / (timeperiod / latent_period) - (s[step] + e[step]) / (1.0 + timeperiod / latent_period) 
        q = -((1.0 + timeperiod / infectivity_period) / (timeperiod * transmission_coeff) * e[step] + (s[step] + e[step]) * i[step]) / ((1.0 + timeperiod / latent_period) * (timeperiod / latent_period))
        e[step + 1] = e[step + 1] = -0.5 * p + math.sqrt(0.25 * p * p - q)
        i[step + 1] = (i[step] + (timeperiod / latent_period) * e[step + 1]) / (1. + timeperiod / infectivity_period)
        s[step + 1] = (s[step]) / (1. + timeperiod * transmission_coeff * i[step + 1])
        r[step + 1] = r[step] + (timeperiod / infectivity_period * i[step + 1])

    return pd.Series(s), pd.Series(e), pd.Series(i), pd.Series(r), times