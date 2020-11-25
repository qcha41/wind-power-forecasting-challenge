import pickle
import pandas as pd
import numpy as np

import tensorflow as tf


def get_last_forecasts(X):
    return get_best_forecasts(X, 1e-16)
    ## Working but old version
    # X = X.copy()
    # X = X.sort_values(by=['update_delay'])
    # X = X.drop_duplicates(subset=['ID', 'WF', 'forecasting_time', 'predictor'],keep='first')
    # X = X.set_index(['ID', 'WF', 'forecasting_time', 'predictor'])
    # X = X.update_value
    # X = X.unstack(level=-1)
    # return X


def get_best_forecasts(X, forecast_memory):
    X = X.copy()
    X.loc[:, 'memory_weight'] = forecast_memory ** X.update_delay
    X.loc[:, 'update_value_weighted'] = X.memory_weight * X.update_value
    gb = X.groupby(['ID', 'WF', 'forecasting_time', 'predictor'])
    X = gb.update_value_weighted.sum() / gb.memory_weight.sum()
    X = X.unstack(level=-1)
    return X


def windowed_dataset(X, Y, window_size, shuffle_buffer):  # , window_size):  # batch_size, shuffle_buffer):
    # series = tf.expand_dims(series, axis=-1)

    # Load features into a tensorflow dataset
    features_dataset = tf.data.Dataset.from_tensor_slices(X)
    # for val in features_dataset:
    #     print(val.numpy().shape) # val : datapoint


    # Make training windows of fixed size
    features_dataset = features_dataset.window(window_size, shift=1, drop_remainder=True)
    # print(len(features_dataset)) # dataset : iterator of # window_datasets of <window_size> datapoints
    # for window_dataset in features_dataset:
    #     print(len(window_dataset)) # window_dataset : iterator of <window_size> datapoints
    #     for val in window_dataset:
    #         print(val.numpy().shape) # val : datapoint
    #     print()
    #     break
    #     break

    # Simply transform training windows into batches
    features_dataset = features_dataset.flat_map(lambda w: w.batch(window_size))
    # print(features_dataset) # dataset : iterator of # individual batches of <window_size> datapoints
    # for ind_batch in features_dataset:
    #     print(ind_batch.numpy().shape) # window : individual batch of <window_size> datapoints
    print(len(list(features_dataset)))
    # Load labels into a tensorflow dataset
    Y = Y[window_size:] # Discard the (window_size-1) first labels as we won't learn to predict them
    labels_dataset = tf.data.Dataset.from_tensor_slices(Y)
    print(len(labels_dataset))
    raise
    # for val in features_dataset:
    #     print(val.numpy().shape) # val : datapoint

    # Shuffle the individual batches
    dataset = dataset.shuffle(shuffle_buffer)

    # Make mini-batches
    dataset = dataset.batch(batch_size)
    # print(dataset) # dataset : iterator of # mini-batches of <batch_size> individual batches of <window_size> datapoints
    # for mini_batch in dataset:
    #     print(mini_batch.numpy().shape) # mini_batch : mini-batch of <batch_size> individual batches of <window_size> datapoints

    # Always prepare 1 mini-batch in advance to be ready when asked
    dataset = dataset.prefetch(1)

    return dataset

if __name__ == '__main__':

    # Hyper-parameters
    forecast_memory = 0.9  # In ]0,1] (0 excluded be it keep only forecasts made a the same time as the forecasting time)
    window_size = 48  # In hours
    shuffle_buffer = 10000
    batch_size = 256
    split_valid = 0.1

    # Load data
    X = pickle.load(open("data/X_train_reshaped.p", "rb"))
    X_test = pickle.load(open("data/X_test_reshaped.p", "rb"))
    Y = pd.read_csv("data/Y_train_sl9m6Jh.csv")

    # Process data
    X = get_best_forecasts(X, forecast_memory)
    X_test = get_best_forecasts(X_test, forecast_memory)
    for X_matrix in [X, X_test]:
        for NWP_id in set([n.split('_')[0] for n in X_matrix.columns]):
            X_matrix.loc[:, NWP_id + '_WS'] = np.sqrt(X_matrix[NWP_id + '_U'] ** 2 + X_matrix[NWP_id + '_V'] ** 2)
            X_matrix.loc[:, NWP_id + '_WD'] = np.arctan2(X_matrix[NWP_id + '_U'], X_matrix[NWP_id + '_V'])
    Y.set_index('ID', inplace=True)

    # Restricting data on WF1 for now
    X = X.xs(1, level=1)
    X_test = X_test.xs(1, level=1)
    Y = Y.loc[X.index.get_level_values(0)]

    # Transform data into numpy arrays
    time = np.array(X.index.get_level_values(0)).squeeze()
    time_test = np.array(X_test.index.get_level_values(0)).squeeze()
    X = np.array(X)
    X_test = np.array(X_test)
    Y = np.array(Y).squeeze()

    # Split train / valid data
    split_valid_id = int(round(len(X) * (1 - split_valid)))
    time_train = time[:split_valid_id]
    X_train = X[:split_valid_id]
    Y_train = Y[:split_valid_id]
    time_valid = time[split_valid_id:]
    X_valid = X[split_valid_id:]
    Y_valid = Y[split_valid_id:]

    # Make windows dataset
    X_train = windowed_dataset(X_train, Y_train, window_size, shuffle_buffer)

    #
    #
    # # Model
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Normalize(),
    #     tf.keras.layers.SimpleRNN(40),
    #     tf.keras.layers.Dense(1),
    # ])
    #
    # model.compile()
    #
    # model.fit()
