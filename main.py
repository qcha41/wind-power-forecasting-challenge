import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def load_data():
    X = pd.read_csv(f'data/X_train_v2.csv', index_col=0)
    Y = pd.read_csv("data/Y_train_sl9m6Jh.csv", index_col=0)
    X_test = pd.read_csv(f'data/X_test_v2.csv', index_col=0)

    # X = pickle.load(open("data/X_train_reshaped.p", "rb"))
    # Y = pd.read_csv("data/Y_train_sl9m6Jh.csv")
    # X_test = pickle.load(open("data/X_test_reshaped.p", "rb"))

    return X, Y, X_test


def restrict_data(X, Y, X_test, WF_num):
    X = X[X.WF == f'WF{WF_num}'].drop(columns=['WF'])
    Y = Y.loc[X.index]
    X_test = X_test[X_test.WF == f'WF{WF_num}'].drop(columns=['WF'])

    return X, Y, X_test


def get_last_forecasts(time, forecasts):
    return get_best_forecasts(time, forecasts, 1e-16)


def get_best_forecasts(time, forecasts, forecast_memory):
    def get_run_infos(run_name):
        predictor = '_'.join(run_name.split('_')[::3])
        run_day_offset_str = run_name.split('_')[2].strip('D')
        run_day_offset = dt.timedelta(days=int(run_day_offset_str) if run_day_offset_str != '' else 0)
        run_time = dt.time(hour=int(run_name.split('_')[1].strip('h')))
        return predictor, run_day_offset, run_time

    def get_update_time(time, run_day_offset, run_time):
        return dt.datetime.combine((time + run_day_offset).date(), run_time)

    def reshape_run_forecasts(time, run):
        run_name = run.name
        run = run.to_frame()
        run.rename(columns={run_name: 'value'}, inplace=True)
        run.loc[:, 'ID'] = run.index.values
        predictor, run_day_offset, run_time = get_run_infos(run_name)
        run.loc[:, 'predictor'] = predictor
        update_time = time.apply(get_update_time, args=(run_day_offset, run_time))
        run.loc[:, 'delay'] = (time - update_time) / pd.Timedelta('1h')
        run.dropna(subset=['value'], inplace=True)
        return run

    forecasts = pd.concat([reshape_run_forecasts(time, forecasts[run_name])
                           for run_name in tqdm(forecasts.columns)], ignore_index=True)
    forecasts.loc[:, 'weight'] = forecast_memory ** forecasts.delay
    forecasts.loc[:, 'value_weighted'] = forecasts.weight * forecasts.value
    gb = forecasts.groupby(['ID', 'predictor'])
    forecasts = gb.value_weighted.sum() / gb.weight.sum()
    forecasts = forecasts.unstack(level=-1)

    return forecasts


def interpolate_missing_values(data):
    if data.isna().sum() > 0:
        sub_data = data.dropna()
        f = interp1d(sub_data.index, sub_data, fill_value="extrapolate")
        data.loc[:] = data.index.map(f)
    return data


def calculate_wind_speed_direction(forecasts):
    for NWP_id in set([predictor.split('_')[0] for predictor in forecasts.columns]):
        # Wind speed
        forecasts.loc[:, NWP_id + '_WS'] = np.sqrt(forecasts[NWP_id + '_U'] ** 2 + forecasts[NWP_id + '_V'] ** 2)
        # Wind direction (angle)
        forecasts.loc[:, NWP_id + '_WD'] = np.arctan2(forecasts[NWP_id + '_U'], forecasts[NWP_id + '_V'])
    return forecasts


def normalize_data(forecasts, forecasts_test):
    forecasts_all = pd.concat([forecasts, forecasts_test])
    for predictor in forecasts.columns:
        mean = forecasts_all[predictor].mean()
        std = forecasts_all[predictor].std()
        forecasts.loc[:, predictor] = (forecasts[predictor] - mean) / std
        forecasts_test.loc[:, predictor] = (forecasts_test[predictor] - mean) / std
    return forecasts, forecasts_test


def windowed_dataset(forecasts, production, window_size, batch_size,
                     shuffle_buffer):  # , window_size):  # batch_size, shuffle_buffer):

    # Forecasts
    forecasts = tf.expand_dims(forecasts, axis=-1)
    forecasts_dataset = tf.data.Dataset.from_tensor_slices(forecasts)
    forecasts_dataset = forecasts_dataset.window(window_size, shift=1,
                                                 drop_remainder=True)  # Make training windows of fixed size
    forecasts_dataset = forecasts_dataset.flat_map(
        lambda w: w.batch(window_size, drop_remainder=True))  # Simply transform training windows into batches

    # Production
    production = tf.expand_dims(production, axis=-1)
    production_dataset = tf.data.Dataset.from_tensor_slices(production)

    # Merge forecasts and production
    dataset = tf.data.Dataset.zip(
        (forecasts_dataset, production_dataset))  # Fuse forecasts and productions datasets together
    # dataset = dataset.shuffle(shuffle_buffer)  # Shuffle the individual batches*
    dataset = dataset.batch(batch_size)  # Make mini-batches of <batch_size> individual batches
    dataset = dataset.prefetch(1)  # Always preload 1 mini-batch in advance to be ready to consume data

    return dataset


if __name__ == '__main__':
    # Hyper-parameters declaration
    forecast_memory = 0.9  # In ]0,1] (0 excluded be it keep only forecasts made a the same time as the forecasting time)
    window_size =  48  # In hours
    shuffle_buffer = 10000
    batch_size = 256
    split_train = 0.85
    WF_num = 1

    # Data loading
    X, Y, X_test = load_data()
    X, Y, X_test = restrict_data(X, Y, X_test, WF_num)

    # Split time / forecasts / production
    time = pd.to_datetime(X['Time'], format='%d/%m/%Y %H:%M')
    forecasts = X.drop(columns=['Time'])
    production = Y
    time_test = pd.to_datetime(X_test['Time'], format='%d/%m/%Y %H:%M')
    forecasts_test = X_test.drop(columns=['Time'])

    # Compute best forecasts
    forecasts = get_best_forecasts(time, forecasts, forecast_memory)
    forecasts_test = get_best_forecasts(time_test, forecasts_test, forecast_memory)

    # Fill missing values with linear interpolation
    forecasts = forecasts.apply(interpolate_missing_values)
    forecasts_test = forecasts_test.apply(interpolate_missing_values)

    # Feature engineering
    forecasts = calculate_wind_speed_direction(forecasts)
    forecasts_test = calculate_wind_speed_direction(forecasts_test)

    # Normalize data
    forecasts, forecasts_test = normalize_data(forecasts, forecasts_test)

    # Split train / valid data
    nb_train_examples = int((len(forecasts) - window_size) * split_train)
    forecasts_train = forecasts.iloc[:nb_train_examples]
    production_train = production.iloc[window_size - 1:nb_train_examples]
    forecasts_valid = forecasts.iloc[nb_train_examples - (window_size - 1):]
    production_valid = production.iloc[nb_train_examples:]

    # Make windows datasets
    dataset_train = windowed_dataset(forecasts_train, production_train, window_size, batch_size, shuffle_buffer)
    dataset_valid = windowed_dataset(forecasts_valid, production_valid, window_size, batch_size, shuffle_buffer)

    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(40, input_shape=[window_size, forecasts.shape[1], 1]),
        tf.keras.layers.Dense(1),
    ])

    # Compile and run model
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=tf.keras.metrics.MeanAbsolutePercentageError())

    model.summary()
    history = model.fit(dataset_train,dataset_valid,validation_data=dataset_valid,epochs=10)

