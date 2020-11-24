import pickle
import pandas as pd

# Hyper-parameters
forecast_memory = 0.9

def get_last_forecasts(X) :
    X = X.copy()
    X = X.sort_values(by=['update_delay'])
    X = X.drop_duplicates(subset=['ID', 'WF', 'forecasting_time', 'predictor'],keep='first')
    X = X.set_index(['ID', 'WF', 'forecasting_time', 'predictor'])
    X = X.update_value
    X = X.unstack(level=-1)
    return X

def get_best_forecasts(X) :
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
X_train = get_last_forecasts(X_train)



