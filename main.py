import pandas as pd
import datetime as dt
#import tensorflow as tf

X_train = pd.read_csv('data/X_train_v2.csv',index_col=0)
X_train['Time'] = pd.to_datetime(X_train['Time'], format='%d/%m/%Y %H:%M')


# Split data per WF
X = {}
for wf_id in range(1,7) :
    X[wf_id] = X_train[X_train.WF==f'WF{wf_id}']
    X[wf_id].drop(columns=['WF'],inplace=True)
    print(X[wf_id].head())

# Produce whether matrix

def get_run_datetime(run_name,current_datetime) :
    day_offset_str = run_name.split('_')[2].strip('D')
    day_offset = int(day_offset_str) if day_offset_str != '' else 0
    run_date = (current_datetime + dt.timedelta(days=day_offset)).date()
    run_time = dt.time(hour=int(run_name.split('_')[1].strip('h')))
    run_datetime = dt.datetime.combine(run_date,run_time)
    return run_name,current_datetime,run_datetime


for name in [a for a in X[1].columns if a != 'Time'] :
    print(get_run_datetime(name,X[1].iloc[0].Time))