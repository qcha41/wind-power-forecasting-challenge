import pandas as pd
import datetime as dt
import pickle
#import tensorflow as tf

forecast_memory = 0.9
i = 0

def get_best_weather_forecast_matrix(X_train) :

    def get_run_delay_hours(run_name, target_datetime):
        day_offset_str = run_name.split('_')[2].strip('D')
        day_offset = int(day_offset_str) if day_offset_str != '' else 0
        run_date = (target_datetime + dt.timedelta(days=day_offset)).date()
        run_time = dt.time(hour=int(run_name.split('_')[1].strip('h')))
        run_datetime = dt.datetime.combine(run_date, run_time)
        run_delay_hours = (target_datetime - run_datetime).total_seconds() / 3600
        return run_delay_hours

    def process_multiple_weather_forecasts(s):
        global i
        i+=1
        print(i)
        target_datetime = s.name[1]
        s.dropna(inplace=True)

        forecasts = pd.DataFrame()
        for run_name in s.index:
            delay_hours = get_run_delay_hours(run_name, target_datetime)
            if delay_hours > 0:
                forecasts = forecasts.append({'name': '_'.join(run_name.split('_')[::3]),
                                              'delay_hours': delay_hours,
                                              'value': s[run_name]}, ignore_index=True)
        forecasts.sort_values(by='name', inplace=True)
        forecasts['weight'] = forecast_memory ** forecasts.delay_hours
        forecasts['weighted_value'] = forecasts.value * forecasts.weight
        g_forecasts = forecasts.groupby(['name'])
        return g_forecasts.weighted_value.sum() / g_forecasts.weight.sum()
        # return g_forecasts.value.mean()

    X_train = X_train.copy()
    X_train['WF'] = X_train.WF.apply(lambda x : int(x.strip('WF')))
    X_train['Time'] = pd.to_datetime(X_train['Time'], format='%d/%m/%Y %H:%M')
    X_train.set_index(['WF', 'Time'],inplace=True)
    X_train = X_train.apply(process_multiple_weather_forecasts,axis=1)

    return X_train

for dataset in ['train','test'] :
    X = pd.read_csv(f'data/X_{dataset}_v2.csv',index_col=0)
    X = get_best_weather_forecast_matrix(X)
    pickle.dump(X, open( f"data/X_{dataset}_processed.p", "wb"))

#print(get_weather_forecast_matrix(X[1].sample().squeeze()))
#data_list[sub_data.Time].name = sub_data.Time
#df = pd.DataFrame(X[1].iloc[0])
#df.columns = ['Value']
#df.name = df.loc['Time','Value']

# Split data per WF
#X = {}
#for wf_id in range(1,7) :
#    X[wf_id] = X_train[X_train.WF==f'WF{wf_id}']
#    X[wf_id].drop(columns=['WF'],inplace=True)
#    print(X[wf_id].head())