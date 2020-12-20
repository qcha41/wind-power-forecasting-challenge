import pandas as pd
import datetime as dt
import numpy as np


def get_last_forecasts(forecasts):
    return get_best_forecasts(forecasts, 1e-16)


def get_best_forecasts(data, forecast_memory):

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
    forecasts = pd.concat([reshape_run_data(data.loc[:,['Time',col]])
                           for col in data.columns if col.startswith('NWP')], ignore_index=True)

    # Computing weight and weighted_value
    forecasts.loc[:, 'weight'] = forecast_memory ** forecasts.delay
    forecasts.loc[:, 'weighted_value'] = forecasts.weight * forecasts.value

    # Computing best forecasts
    gb = forecasts.groupby(['ID', 'predictor'])
    forecasts = gb.weighted_value.sum() / gb.weight.sum()
    forecasts = forecasts.unstack(level=-1)

    return forecasts


def calculate_wind_speed_direction(data):

    return forecasts


if __name__ == '__main__':

    # Processing parameters
    forecast_memory = 0.8  # In ]0,1] (0 excluded because it keep only forecasts made a the same time as the forecasting time which may not exist)

    # Load data from file
    X_train = pd.read_csv(f'data/X_train_v2.csv', index_col=0)
    X_test = pd.read_csv(f'data/X_test_v2.csv', index_col=0)
    Y_train = pd.read_csv("data/Y_train_sl9m6Jh.csv", index_col=0)

    # Make global dataframe
    data = pd.concat([X_train, X_test])
    data.loc[:, 'Time'] = pd.to_datetime(data['Time'], format='%d/%m/%Y %H:%M')
    data.loc[:, 'WF'] = data.WF.apply(lambda x: int(x.strip('WF')))
    data.loc[Y_train.index, 'Production'] = Y_train

    # Replace NWP runs forecasts by the best computed forecasts
    not_NWP_cols = [col for col in data.columns if not col.startswith('NWP')]
    data = pd.concat([data[not_NWP_cols], get_best_forecasts(data, forecast_memory)], axis=1)

    # Interpolate missing values within each Wind farms
    NWP_cols = [col for col in data.columns if col.startswith('NWP')]
    data.loc[:,NWP_cols] = data.groupby('WF')[NWP_cols].apply(lambda group: group.interpolate(method='linear',
                                                                                              limit_direction='both'))

    # Append features
    for NWP_num in set([predictor.split('_')[0] for predictor in NWP_cols]):
        data.loc[:, NWP_num + '_WS'] = np.sqrt(data[NWP_num + '_U'] ** 2 + data[NWP_num + '_V'] ** 2) # Wind speed
        data.loc[:, NWP_num + '_WD'] = np.arctan2(data[NWP_num + '_U'], data[NWP_num + '_V']) # Wind direction (angle)


    # Normalize data
    forecasts = data.loc[:, NWP_cols]
    data.loc[:, NWP_cols] = (forecasts - forecasts.mean()) / forecasts.std()

    # Save new dataframe
    data.to_csv('data/data_processed.csv')