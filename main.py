import wf_process
import pandas as pd
import utilities
import seaborn as sns
import matplotlib.pyplot as plt
plt.ioff()
sns.set_theme(style="darkgrid")

def load_data():

    # Load data from file
    X_train = pd.read_csv(f'data/X_train_v2.csv', index_col=0)
    X_test = pd.read_csv(f'data/X_test_v2.csv', index_col=0)
    Y_train = pd.read_csv("data/Y_train_sl9m6Jh.csv", index_col=0)

    # Make global dataframe
    data = pd.concat([X_train,X_test])
    data.loc[:, 'Time'] = pd.to_datetime(data['Time'], format='%d/%m/%Y %H:%M')
    data.loc[:, 'WF'] = data.WF.apply(lambda x: int(x.strip('WF')))
    data.loc[Y_train.index, Y_train.columns[0]] = Y_train

    return data


def restrict_data(data, WF_num):

    data_wf = data[data.WF == WF_num]
    data_wf.drop(columns=['WF'],inplace=True)

    return data_wf


if __name__=='__main__' :

    # Data loading
    data = load_data()

    run_manager = utilities.RunManager('test')

    for WF_num in data.WF.unique() :

        # Restrict data to selected WF
        data_wf = restrict_data(data, WF_num)

        # Train and predict model
        output = wf_process.train_predict(data_wf)
        run_manager.add_predictions(WF_num, output)

        # Plots
        run_manager.plot_learning_curves()
        run_manager.plot_predictions()
        break

    # Merge all predictions in a file
    run_manager.output_predictions()

