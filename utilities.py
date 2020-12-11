import datetime as dt
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class RunManager:
    main_folder_path = os.path.realpath('./runs')

    def __init__(self, name):
        self.name = name
        self.time = dt.datetime.now()
        self.folder_name = self.time.strftime('%Y%m%d_%H%M%S') + '_' + self.name.strip(' ')
        self.folder_path = os.path.join(self.main_folder_path, self.folder_name)

        # Initialize folder
        if not os.path.exists(self.main_folder_path): os.mkdir(self.main_folder_path)
        os.mkdir(self.folder_path)

        # Backup files
        backup_folder_path = os.path.join(self.folder_path, 'backup')
        os.mkdir(backup_folder_path)
        for file_path in [os.path.join('.', f) for f in os.listdir('.') if f.endswith('.py')]:
            shutil.copy(file_path, backup_folder_path)

        # For upcoming results
        self.data = {}

    def add_predictions(self, WF_num, data):
        self.data[WF_num] = data

    def output_predictions(self):
        predictions = pd.concat([d['production'][d['production'].dataset=='test'].predict for d in self.data.values()])
        predictions.to_csv(os.path.join(self.folder_path, 'Y_test.csv'), float_format='%.2f')

    def save_plot(self, fig, title):
        fig.suptitle(title)
        fig.savefig(os.path.join(self.folder_path, f'{title}.jpg'))

    def plot_learning_curves(self):
        for WF_num in self.data.keys():
            fig = plt.figure()
            ax = sns.lineplot(data=self.data[WF_num]['history'])
            ax.set(xlabel='epoch', ylabel='loss')
            self.save_plot(fig, f'{WF_num}_learning_curve')
            plt.close(fig)

    def plot_predictions(self):
        for WF_num in self.data.keys():
            fig = plt.figure()
            d = self.data[WF_num]['production'].set_index('time')
            d = d[d.dataset!='test'][['predict','true']]
            ax= sns.lineplot(data=d)
            self.save_plot(fig, f'{WF_num}_data_wf')
            ax.set_xscale()
            plt.close(fig)
