import pandas as pd
import datetime as dt
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt, mpld3
import mlflow
import tempfile
import os

# sns.set(rc={'figure.figsize': (11.7, 8.27)})
temp_folder = tempfile.mkdtemp()
shuffle_buffer = 10000


def load_data():
    # Load raw data from file
    X_train = pd.read_csv(f'data/X_train_v2.csv', index_col=0)
    X_test = pd.read_csv(f'data/X_test_v2.csv', index_col=0)
    Y_train = pd.read_csv("data/Y_train_sl9m6Jh.csv", index_col=0)

    # Make global dataframe with it
    df = pd.concat([X_train, X_test])
    df.loc[:, 'Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M')
    df.loc[:, 'WF'] = df.WF.apply(lambda x: int(x.strip('WF')))
    df.loc[Y_train.index, 'Production'] = Y_train

    return df


def calculate_best_forecasts(df, forecast_memory):
    def get_run_infos(run_name):
        predictor = '_'.join(run_name.split('_')[::3])
        run_day_offset_str = run_name.split('_')[2].strip('D')
        run_day_offset = int(run_day_offset_str) if run_day_offset_str != '' else 0
        run_time = int(run_name.split('_')[1].strip('h'))
        timedelta = dt.timedelta(days=run_day_offset, hours=run_time)
        return predictor, timedelta

    def reshape_run_data(run_data):
        # Preparation
        run_name = [col for col in run_data.columns if col.startswith('NWP')][0]

        # Computing predictor and delay
        predictor, timedelta = get_run_infos(run_name)
        run_data.loc[:, 'predictor'] = predictor
        update_time = run_data.Time.dt.normalize() + timedelta
        run_data.loc[:, 'delay'] = (run_data.Time - update_time) / pd.Timedelta('1h')

        # Cleaning
        run_data.rename(columns={run_name: 'value'}, inplace=True)
        run_data.drop(columns='Time', inplace=True)
        run_data.dropna(subset=['value'], inplace=True)
        run_data.reset_index(inplace=True)

        return run_data

    # Reshape and concatenate each NWP_run data ( index=[ID,predictor]] , columns=[delay,value] )
    best_forecasts = pd.concat([reshape_run_data(df.loc[:, ['Time', col]])
                                for col in df.columns if col.startswith('NWP')], ignore_index=True)

    # Computing weight and weighted_value
    best_forecasts.loc[:, 'weight'] = forecast_memory ** best_forecasts.delay
    best_forecasts.loc[:, 'weighted_value'] = best_forecasts.weight * best_forecasts.value

    # Computing best forecasts
    gb = best_forecasts.groupby(['ID', 'predictor'])
    best_forecasts = gb.weighted_value.sum() / gb.weight.sum()
    best_forecasts = best_forecasts.unstack(level=-1)

    # Replace initial NWP runs forecasts by the best computed forecasts
    not_NWP_cols = [col for col in df.columns if not col.startswith('NWP')]
    df = pd.concat([df[not_NWP_cols], best_forecasts], axis=1)

    return df


def get_NWP_cols(df):
    return [col for col in df.columns if col.startswith('NWP')]


def interpolate_nans(df):
    # Interpolate missing values within each Wind farms
    NWP_cols = get_NWP_cols(df)
    gb = df.groupby('WF')[NWP_cols]
    df.loc[:, NWP_cols] = gb.apply(lambda group: group.interpolate(method='linear', limit_direction='both'))
    return df


def augment_data(df):
    # Append wind speed and direction
    NWP_cols = get_NWP_cols(df)
    for NWP_num in set([col.split('_')[0] for col in NWP_cols]):
        df.loc[:, NWP_num + '_WS'] = np.sqrt(df[NWP_num + '_U'] ** 2 + df[NWP_num + '_V'] ** 2)  # Wind speed
        df.loc[:, NWP_num + '_WD'] = np.arctan2(df[NWP_num + '_U'], df[NWP_num + '_V'])  # Wind direction (angle)

    # Append features mean across prediction sources
    NWP_cols = get_NWP_cols(df)
    for NWP_var in set([col.split('_')[1] for col in NWP_cols]):
        predictors = [col for col in NWP_cols if col.split('_')[1] == NWP_var]
        if len(predictors) > 1:
            df.loc[:, 'NWP0_' + NWP_var] = df[predictors].mean(axis=1)

    return df


def normalize_data(df):
    NWP_cols = get_NWP_cols(df)
    # CLCT variable is a percentage --> divide it by 100
    NWP_cols_CLCT = [col for col in NWP_cols if 'CLCT' in col]
    df.loc[:, NWP_cols_CLCT] /= 100
    # Other variables --> centered and normalized
    NWP_cols_noCLCT = [col for col in NWP_cols if 'CLCT' not in col]
    data = df.loc[:, NWP_cols_noCLCT]
    df.loc[:, NWP_cols_noCLCT] = (data - data.mean()) / data.std()
    return df


def extract_wf_data(df, wf_num):
    return df.query(f'WF=={wf_num}').drop(columns=['WF'])


def extract_period_data(df_wf, i_ini, i_end, window_size):
    NWP_cols = get_NWP_cols(df_wf)
    t = df_wf.iloc[i_ini:i_end].Time
    x = df_wf.iloc[i_ini - (window_size - 1):i_end][NWP_cols]
    y = df_wf.iloc[i_ini:i_end].Production
    return t, x, y


def get_windowed_dataset(x, y, window_size, batch_size, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)  # Make training windows of fixed size
    dataset = dataset.flat_map(lambda w: w.batch(window_size, drop_remainder=True))  # Transform windows in batches
    if y is not None:
        y_dataset = tf.data.Dataset.from_tensor_slices(y)
        dataset = tf.data.Dataset.zip((dataset, y_dataset))  # Merge forecasts and production
    if shuffle is True:
        dataset = dataset.shuffle(shuffle_buffer)  # Shuffle the individual batches
    dataset = dataset.batch(batch_size)  # Make mini-batches of <batch_size> individual batches
    dataset = dataset.prefetch(1)  # Always preload 1 mini-batch in advance to be ready to consume data
    return dataset


def split_holdout_validation(df_wf, split, window_size):
    nb_labelled_data = len(df_wf.Production.dropna())
    i_split = int(nb_labelled_data * split)
    t_train, x_train, y_train = extract_period_data(df_wf, window_size - 1, i_split, window_size)
    t_valid, x_valid, y_valid = extract_period_data(df_wf, i_split, nb_labelled_data, window_size)
    return t_train, x_train, y_train, t_valid, x_valid, y_valid


def split_forward_chaining_validation(df_wf, valid_size, nb_valid, window_size):
    nb_labelled_data = len(df_wf.Production.dropna())
    sets = []
    for i in range(nb_valid):
        i_valid_ini = int((1 - (nb_valid - i) * valid_size) * nb_labelled_data)
        i_valid_fin = int((1 - (nb_valid - i - 1) * valid_size) * nb_labelled_data)
        t_train, x_train, y_train = extract_period_data(df_wf, window_size - 1, i_valid_ini, window_size)
        t_valid, x_valid, y_valid = extract_period_data(df_wf, i_valid_ini, i_valid_fin, window_size)
        sets.append((t_train, x_train, y_train, t_valid, x_valid, y_valid))
    return sets


def get_train_dataset(df_wf, window_size):
    nb_labelled_data = len(df_wf.Production.dropna())
    t, x, y = extract_period_data(df_wf, window_size - 1, nb_labelled_data, window_size)
    return t, x, y


def get_test_dataset(df_wf, window_size):
    nb_labelled_data = len(df_wf.Production.dropna())
    t, x, y_fake = extract_period_data(df_wf, nb_labelled_data, None, window_size)
    return t, x


def save_plot(fig, file_name):
    file_path = os.path.join(temp_folder, file_name + '.html')
    mpld3.save_html(plt.gcf(), file_path)
    mlflow.log_artifact(file_path)


def plot_learning_curves(history):
    plt.figure()
    for key in history.history.keys():
        sns.lineplot(x=range(len(history.history[key])), y=history.history[key], label=key)
    save_plot(plt.gcf(), 'learning_curves')


def predict(model, dataset, time):
    return pd.Series(model.predict(dataset).squeeze(), index=time.index)


def plot_predictions(time, y_true, y_predict, title):
    plt.figure(figsize=(15, 6))
    sns.lineplot(x=time, y=y_predict, label='y_predict')
    if y_true is not None:
        sns.lineplot(x=time, y=y_true, label='y_true')
    plt.title(title)
    save_plot(plt.gcf(), f'{title}_predictions')


def save_predictions(predictions):
    predictions = pd.concat([p for p in predictions]) \
        .sort_index() \
        .rename('Production') \
        .round(decimals=2)
    file_path = os.path.join(temp_folder, 'predictions.csv')
    predictions.to_csv(file_path)
    mlflow.log_artifact(file_path)


def get_mean_std_metrics(metrics):
    cross_metrics = {}
    for key in metrics[0].keys():
        values = [history[key][-1] for history in metrics]
        cross_metrics[f'{key}'] = np.mean(values)
        cross_metrics[f'{key}_std'] = np.std(values)
    return cross_metrics
