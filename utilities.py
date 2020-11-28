import datetime as dt
import os
import shutil
import pandas as pd


class RunManager :

    main_folder_path = os.path.realpath('./runs')

    def __init__(self,name):
        self.name = name
        self.time = dt.datetime.now()
        self.folder_name = self.time.strftime('%Y%m%d_%H%M%S')+'_'+self.name.strip(' ')
        self.folder_path = os.path.join(self.main_folder_path, self.folder_name)

        # Initialize folder
        if not os.path.exists(self.main_folder_path): os.mkdir(self.main_folder_path)
        os.mkdir(self.folder_path)

        # Backup files
        backup_folder_path = os.path.join(self.folder_path,'backup')
        os.mkdir(backup_folder_path)
        for file_path in [os.path.join('.',f) for f in os.listdir('.') if f.endswith('.py')] :
            shutil.copy(file_path,backup_folder_path)

        # For upcoming results
        self.predictions_list = []

    def add_predictions(self, predictions):
        self.predictions_list.append(predictions[predictions.true.isna()].predict)

    def save_plot(self, fig, title):
        fig.suptitle(title)
        fig.savefig(os.path.join(self.folder_path,f'{title}.jpg'))

    def output_predictions(self):

        predictions = pd.concat(self.predictions_list)
        predictions.to_csv(os.path.join(self.folder_path,'Y_test.csv'),float_format='%.2f')