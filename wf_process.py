import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def get_last_forecasts(forecasts):
    return get_best_forecasts(forecasts, 1e-16)


def get_best_forecasts(time, forecasts, forecast_memory):

    def get_run_infos(run_name):
        predictor = '_'.join(run_name.split('_')[::3])
        run_day_offset_str = run_name.split('_')[2].strip('D')
        run_day_offset = int(run_day_offset_str) if run_day_offset_str != '' else 0
        run_time = int(run_name.split('_')[1].strip('h'))
        timedelta = dt.timedelta(days=run_day_offset,hours=run_time)
        return predictor, timedelta

    def reshape_run_forecasts(time, rf):

        run_name = rf.name
        rf = rf.to_frame()
        rf.rename(columns={run_name: 'value'}, inplace=True)
        predictor, timedelta = get_run_infos(run_name)
        rf.loc[:, 'predictor'] = predictor
        update_time = time.dt.normalize() + timedelta
        rf.loc[:, 'delay'] = (time - update_time) / pd.Timedelta('1h')
        rf.dropna(subset=['value'], inplace=True)
        rf.reset_index(inplace=True)

        return rf

    forecasts = pd.concat([reshape_run_forecasts(time, forecasts[run_name])
                           for run_name in tqdm(forecasts.columns)],ignore_index=True)
    forecasts.loc[:, 'weight'] = forecast_memory ** forecasts.delay
    forecasts.loc[:, 'value_weighted'] = forecasts.weight * forecasts.value
    gb = forecasts.groupby(['ID', 'predictor'])
    forecasts = gb.value_weighted.sum() / gb.weight.sum()
    forecasts = forecasts.unstack(level=-1)

    return forecasts


def calculate_wind_speed_direction(forecasts):
    for NWP_id in set([predictor.split('_')[0] for predictor in forecasts.columns]):
        # Wind speed
        forecasts.loc[:, NWP_id + '_WS'] = np.sqrt(forecasts[NWP_id + '_U'] ** 2 + forecasts[NWP_id + '_V'] ** 2)
        # Wind direction (angle)
        forecasts.loc[:, NWP_id + '_WD'] = np.arctan2(forecasts[NWP_id + '_U'], forecasts[NWP_id + '_V'])
    return forecasts


def windowed_dataset(forecasts, window_size, batch_size,
                     shuffle_buffer, production=None):

    # Forecasts
    dataset = tf.data.Dataset.from_tensor_slices(forecasts)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)  # Make training windows of fixed size
    dataset = dataset.flat_map(lambda w: w.batch(window_size, drop_remainder=True))  # Transform windows in batches

    # Production
    if production is not None :
        production_dataset = tf.data.Dataset.from_tensor_slices(production)
        dataset = tf.data.Dataset.zip((dataset, production_dataset))  # Merge forecasts and production
        dataset = dataset.shuffle(shuffle_buffer)  # Shuffle the individual batches

    # Batch and prefetch
    dataset = dataset.batch(batch_size)  # Make mini-batches of <batch_size> individual batches
    dataset = dataset.prefetch(1)  # Always preload 1 mini-batch in advance to be ready to consume data

    return dataset


def train_predict(data) :

    # Hyper-parameters declaration
    forecast_memory = 0.9  # In ]0,1] (0 excluded be it keep only forecasts made a the same time as the forecasting time)
    window_size =  48 # In hours
    shuffle_buffer = 10000
    batch_size = 1000
    split_train = 0.85
    coeff_production = 0.1
    epochs = 3

    data = data.copy()

    # Split forecasts / production
    time = data.pop('Time')
    production = data.pop('Production').to_frame().rename(columns={'Production':'true'})
    forecasts = data


    # Feature engineering
    forecasts = get_best_forecasts(time, forecasts, forecast_memory) # Compute best forecasts
    forecasts.interpolate(method='linear', limit_direction='both', axis=0, inplace=True) # Interpolate missing values
    forecasts = calculate_wind_speed_direction(forecasts) # Compute wind speed and direction based on U, V

    # Normalize data
    forecasts = (forecasts - forecasts.mean()) / forecasts.std()

    # Split train / valid / test data
    train_length = len(production.dropna())
    nb_train_examples = int((train_length - window_size) * split_train)
    production.loc[production.index[window_size - 1:nb_train_examples],'dataset'] = 'train'
    production.loc[production.index[nb_train_examples:train_length], 'dataset'] = 'valid'
    production.loc[production.true.isna(),'dataset'] = 'test'
    forecasts_train = forecasts.iloc[:nb_train_examples]
    forecasts_valid = forecasts.iloc[nb_train_examples - (window_size - 1):train_length]
    forecasts_test = forecasts.iloc[train_length - (window_size - 1):]
    dataset_train = windowed_dataset(forecasts_train, window_size, batch_size, shuffle_buffer,
                                     production=production[production.dataset == 'train'].true)
    dataset_valid = windowed_dataset(forecasts_valid, window_size, batch_size, shuffle_buffer,
                                     production=production[production.dataset == 'valid'].true)
    dataset_test = windowed_dataset(forecasts_test, window_size, batch_size, shuffle_buffer)

    # Define NN model
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(40, input_shape=[window_size, forecasts.shape[1]]),
        tf.keras.layers.Dense(1,activation='relu'),
    ])

    # Compile and run model
    model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  optimizer=tf.keras.optimizers.Adam())
    history = model.fit(dataset_train,validation_data=dataset_valid,epochs=epochs)

    # Predictions
    production.loc[production.dataset=='train', 'predict'] = model.predict(dataset_train).squeeze() / coeff_production
    production.loc[production.dataset=='valid', 'predict'] = model.predict(dataset_valid).squeeze() / coeff_production
    production.loc[production.dataset=='test', 'predict'] = model.predict(dataset_test).squeeze() / coeff_production
    return production, pd.DataFrame(history.history)
