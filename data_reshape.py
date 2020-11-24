import pandas as pd
import pickle
import datetime as dt
from tqdm import tqdm


def get_run_infos(run_name):
    predictor = '_'.join(run_name.split('_')[::3])
    run_day_offset_str = run_name.split('_')[2].strip('D')
    run_day_offset = dt.timedelta(days=int(run_day_offset_str) if run_day_offset_str != '' else 0)
    run_time = dt.time(hour=int(run_name.split('_')[1].strip('h')))
    return predictor, run_day_offset, run_time


def get_update_time(forecasting_time, day_offset, run_time):
    return dt.datetime.combine((forecasting_time + day_offset).date(), run_time)


def reshape_run_data(run):
    run_name = [n for n in run.columns.values if n.startswith('NWP')][0]
    run = run.copy()
    predictor, run_day_offset, run_time = get_run_infos(run_name)
    run.rename(columns={run_name: 'update_value'}, inplace=True)
    run.loc[:, 'ID'] = run.index.values
    run.loc[:, 'predictor'] = predictor
    run.loc[:, 'update_time'] = run.forecasting_time.apply(get_update_time, args=(run_day_offset, run_time))
    run.loc[:, 'update_delay'] = (run.forecasting_time - run.update_time) / pd.Timedelta('1h')
    run.dropna(subset=['update_value'], inplace=True)
    return run


for dataset in ['train','test']:

    # Load data
    X = pd.read_csv(f'data/X_{dataset}_v2.csv', index_col=0)

    # Change columns names
    X.rename(columns={'Time': 'forecasting_time'}, inplace=True)

    # Reformat data
    X['WF'] = X.WF.apply(lambda x: int(x.strip('WF')))
    X['forecasting_time'] = pd.to_datetime(X['forecasting_time'], format='%d/%m/%Y %H:%M')

    # Reshape data
    run_names = [n for n in X.columns.values if n not in ['WF', 'forecasting_time']]
    X = pd.concat([reshape_run_data(X[['WF', 'forecasting_time', run_name]])
                   for run_name in tqdm(run_names)], ignore_index=True)

    # Save new data
    pickle.dump(X, open(f"data/X_{dataset}_reshaped.p", "wb"))
