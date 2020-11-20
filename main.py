import pandas as pd
#import tensorflow as tf

X_train = pd.read_csv('data/X_train_v2.csv',index_col=0)

print(X_train.head())

# Split data per WF
X = {}
for wf_id in range(1,7) :
    X[wf_id] = X_train[X_train.WF==f'WF{wf_id}']
    X[wf_id].drop(columns=['WF'],inplace=True)
    print(X[wf_id].head())