import numpy as np
import pandas as pd
import datetime as dt
from .utilities import get_nwp_cols, TEMP_FOLDER


def load_data():
    """ Returns full raw data into a single Pandas DataFrame """

    # Load raw data from file
    x_train = pd.read_csv(f'./data/X_train_v2.csv', index_col=0)
    x_test = pd.read_csv(f'./data/X_test_v2.csv', index_col=0)
    y_train = pd.read_csv("./data/Y_train_sl9m6Jh.csv", index_col=0)

    # Make global dataframe with it
    df = pd.concat([x_train, x_test])
    df.loc[:, 'Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M')
    df.loc[:, 'WF'] = df.WF.apply(lambda x: int(x.strip('WF')))
    df.loc[y_train.index, 'Production'] = y_train

    return df


def calculate_best_forecasts(df, forecast_memory):
    """ Compute and returns the best forecasts dataframe """

    def get_run_infos(run_name):
        """ Returns predictor name and timedelta from run name"""

        predictor = '_'.join(run_name.split('_')[::3])
        run_day_offset_str = run_name.split('_')[2].strip('D')
        run_day_offset = int(run_day_offset_str) if run_day_offset_str != '' else 0
        run_time = int(run_name.split('_')[1].strip('h'))
        timedelta = dt.timedelta(days=run_day_offset, hours=run_time)
        return predictor, timedelta

    def reshape_run_data(run_data):
        """ Returns run data in a dataframe with predictor name, delay and value columns """

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
    not_nwp_cols = [col for col in df.columns if not col.startswith('NWP')]
    df = pd.concat([df[not_nwp_cols], best_forecasts], axis=1)

    return df


def interpolate_nans(df):
    """ Interpolate missing values within each Wind farms """

    nwp_cols = get_nwp_cols(df)
    gb = df.groupby('WF')[nwp_cols]
    df.loc[:, nwp_cols] = gb.apply(lambda group: group.interpolate(method='linear', limit_direction='both'))
    return df


def append_features(df):
    """ Append wind speed and direction features in the dataframe"""

    nwp_cols = get_nwp_cols(df)
    for NWP_num in set([col.split('_')[0] for col in nwp_cols]):
        df.loc[:, NWP_num + '_WS'] = np.sqrt(df[NWP_num + '_U'] ** 2 + df[NWP_num + '_V'] ** 2)  # Wind speed
        df.loc[:, NWP_num + '_WD'] = np.arctan2(df[NWP_num + '_U'], df[NWP_num + '_V'])  # Wind direction (angle)
    return df


def detect_manual_shutdown(df, std_coeff=1.5):
    """ Append a flag 'manual_shutdown' to data where power production is 0 whereas
    wind speed is strong """

    df = df.copy()
    df['manual_shutdown'] = False
    ws_cols = [col for col in get_nwp_cols(df) if col.endswith('WS')]
    for wf_num in df.WF.unique():
        # For each wind farm, keep only wind speed forecasts, where power production is 0
        df_wf = df.loc[(df.WF == wf_num) & (df.Production == 0), ws_cols]
        # Compute wind speed thresholds per wind speed forecast, based on mean and std of these remaining forecasts
        thresholds = df_wf.mean() + std_coeff*df_wf.std()
        # If a given wind speed value is above threshold, the data at this time is suspected
        suspicion_table = (df_wf - thresholds > 0)
        # If the point is suspected on all forecasts, the WF is declared as manually shutdown for that time
        declared = (suspicion_table.sum(axis=1) == 4)
        df.loc[df_wf.loc[declared].index, 'manual_shutdown'] = True
    return df


def mean_data(df):
    """ Returns a dataframe containing the mean value for each weather variable across the NWP models """

    nwp_cols = get_nwp_cols(df)
    df_mean = df.drop(columns=nwp_cols)
    for NWP_var in set([col.split('_')[1] for col in nwp_cols]):
        predictors = [col for col in nwp_cols if col.split('_')[1] == NWP_var]
        df_mean.loc[:, NWP_var] = df[predictors].mean(axis=1)
    return df_mean


def normalize_data(df):
    """ Normalize features according to their types """

    df = df.copy()
    nwp_cols = get_nwp_cols(df)

    # CLCT variable is a percentage --> divide it by 100
    nwp_cols_clct = [col for col in nwp_cols if 'CLCT' in col]
    df.loc[:, nwp_cols_clct] /= 100

    # Other variables --> centered and normalized
    nwp_cols_no_clct = [col for col in nwp_cols if 'CLCT' not in col]
    data = df.loc[:, nwp_cols_no_clct]
    df.loc[:, nwp_cols_no_clct] = (data - data.mean()) / data.std()

    return df


def extract_wf_data(df, wf_num):
    """ Returns the data associated to the given wind farm ID """

    return df.query(f'WF=={wf_num}').drop(columns=['WF'])


def set_disabled_flag(df, inactivity_periods, delay_threshold):
    """ Set the feature 'disabled' to True in the dataset
    for examples in periods of inactivity
    longer than delay_threshold """
    df = df.copy()
    df.loc[:, 'disabled'] = False
    for period in [p for p in inactivity_periods if p['hours'] > delay_threshold]:
        ID = period['ID_ini']
        df.loc[ID:ID + period['hours'], 'disabled'] = True
    return df


class WFData:
    def __init__(self, df, wf_num, window_size):
        self.wf_num = wf_num
        self.df_wf = df[df.WF == self.wf_num].sort_values(by="Time").reset_index()
        self.df_wf_train = self.df_wf.loc[self.df_wf.Production.notna()]
        self.df_wf_test = self.df_wf.loc[self.df_wf.Production.isna()]
        self.window_size = window_size

    def get_train_data(self):
        """ Returns the windowed full training dataset of the associated wind farm """

        index_train = self.df_wf_train.index[self.window_size - 1:].values
        return self._get_windowed_data(index_train)

    def get_test_data(self):
        """ Returns the windowed full test dataset of the associated wind farm """

        index_test = self.df_wf_test.index.values
        return self._get_windowed_data(index_test)

    def _split_train_valid(self, end_train, end_valid):
        """ Internal method that split the full training data of the associated wind farm
        into a training and a validation dataset """

        i_end_train = int(len(self.df_wf_train) * end_train)
        i_end_valid = int(len(self.df_wf_train) * end_valid)
        index_train = self.df_wf_train.index[self.window_size - 1:i_end_train].values
        index_valid = self.df_wf_train.index[i_end_train:i_end_valid].values
        return index_train, index_valid

    def split_train_valid_holdout(self, split):
        """ Split full training dataset into a training and a validation datasets
        using the holdout method (1 training dataset, 1 validation dataset) """

        end_train = split
        end_valid = 1
        index_train, index_valid = self._split_train_valid(end_train, end_valid)
        t_train, x_train, y_train = self._get_windowed_data(index_train)
        t_valid, x_valid, y_valid = self._get_windowed_data(index_valid)
        yield t_train, x_train, y_train, t_valid, x_valid, y_valid

    def split_train_valid_forward_chaining(self, nb_steps, valid_size):
        """ Split full training dataset into a training and a validation datasets
        using the forward chaining method: nb_steps * (1 training dataset, 1 validation dataset)
        with sliding window of size valid_size -- well adapted for time series """

        for n in range(nb_steps):
            end_train = 1 - (nb_steps - n) * valid_size
            end_valid = 1 - (nb_steps - n - 1) * valid_size
            index_train, index_valid = self._split_train_valid(end_train, end_valid)
            t_train, x_train, y_train = self._get_windowed_data(index_train)
            t_valid, x_valid, y_valid = self._get_windowed_data(index_valid)
            yield t_train, x_train, y_train, t_valid, x_valid, y_valid

    def _get_windowed_data(self, index):
        """ Internal method that transforms data of given index into a windowed dataset """

        index = self._remove_manual_shutdown_index(index)
        t = self.df_wf.loc[index].set_index('ID')["Time"]
        x = self.df_wf[get_nwp_cols(self.df_wf)]
        x = np.array([x.loc[i - self.window_size + 1:i].values for i in index])
        y = self.df_wf.loc[index].set_index('ID')["Production"]
        return t, x, y

    def _remove_manual_shutdown_index(self, index):
        """ Removes index that correspond to manual shutdown """

        if 'disabled' in self.df_wf.columns :
            data = self.df_wf.loc[index]
            not_shutdown_data = data[data.disabled == False]
            return not_shutdown_data.index.values
        else :
            return index
