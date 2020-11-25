import pickle
import pandas as pd
import numpy as np

# Hyper-parameters
forecast_memory = 0.9  # In ]0,1] (0 excluded be it keep only forecasts made a the same time as the forecasting time)

def get_last_forecasts(X) :
    return get_best_forecasts(X,1e-16)
    ## Working but old version
    # X = X.copy()
    # X = X.sort_values(by=['update_delay'])
    # X = X.drop_duplicates(subset=['ID', 'WF', 'forecasting_time', 'predictor'],keep='first')
    # X = X.set_index(['ID', 'WF', 'forecasting_time', 'predictor'])
    # X = X.update_value
    # X = X.unstack(level=-1)
    # return X

def get_best_forecasts(X,forecast_memory) :
    X = X.copy()
    X.loc[:, 'memory_weight'] = forecast_memory**X.update_delay
    X.loc[:, 'update_value_weighted'] = X.memory_weight*X.update_value
    gb = X.groupby(['ID','WF','forecasting_time','predictor'])
    X = gb.update_value_weighted.sum()/gb.memory_weight.sum()
    X = X.unstack(level=-1)
    return X

# Load data
X_train = pickle.load(open("data/X_train_reshaped.p", "rb"))
X_test = pickle.load(open("data/X_test_reshaped.p", "rb"))
Y_train = pd.read_csv("data/Y_train_sl9m6Jh.csv")

# Process data
X_train_best = get_best_forecasts(X_train,forecast_memory)
print(X_train_best)