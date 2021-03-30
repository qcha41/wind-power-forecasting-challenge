import os
import mlflow
import numpy as np
import pandas as pd

from .utilities import get_nwp_cols, TEMP_FOLDER


class CrossValidationResults:
    def __init__(self):
        self.results = {}

    def clear(self, wf_num=None):
        if wf_num is None:
            self.results = {}
        elif wf_num in self.results.keys():
            del self.results[wf_num]

    def add(self, wf_num, history):
        if wf_num not in self.results.keys():
            self.results[wf_num] = pd.DataFrame()
        results = {}
        for key in history.history.keys():
            results[key] = history.history[key][-1]
        results['epochs'] = len(history.history[key])
        self.results[wf_num] = self.results[wf_num].append(results, ignore_index=True)

    def stats(self, wf_num):
        return self.results[wf_num].describe().loc[['mean', 'std']]

    def global_stats(self):
        values = np.array([self.stats(wf_num).loc['mean','val_loss'] for wf_num in range(1,7)])
        return {'loss':values.mean(), 'loss_std':values.std()}

    def stats_dict(self, wf_num):
        stats = self.stats(wf_num).transpose()
        stats_dict = stats['mean'].to_dict()
        stats_dict.update(stats['std'].add_suffix('_std').to_dict())
        return stats_dict

    def status(self):
        status = ''
        for wf_num in range(1, 7):
            status += f'WF{wf_num}\n'
            if wf_num in self.results.keys():
                status += ' - Data\n'
                status += str(self.results[wf_num])
                status += '\n - Summary\n'
                status += str(self.stats(wf_num))
            else:
                status += ' - No data\n'
            status += '\n\n'
        return status


def get_mean_std_metrics(metrics):
    cross_metrics = {}
    for key in metrics[0].keys():
        values = [history[key][-1] for history in metrics]
        cross_metrics[f'{key}'] = np.mean(values)
        cross_metrics[f'{key}_std'] = np.std(values)
    return cross_metrics


def save_predictions(predictions):
    """ Save in csv file the given model predictions on test data """

    predictions = pd.concat([p for p in predictions]) \
        .sort_index() \
        .rename('Production') \
        .round(decimals=2)
    save_data(predictions, 'predictions.csv')


def save_data(data, filename):
    """ Save given dataframe into a file and upload it in mlflow"""

    file_path = os.path.join(TEMP_FOLDER, filename)
    data.to_csv(file_path)
    mlflow.log_artifact(file_path)


def save_text(data, filename):
    """ Save given text into a file and upload it in mlflow"""

    file_path = os.path.join(TEMP_FOLDER, filename)
    with open(file_path, 'w') as file:
        file.write(data)
    mlflow.log_artifact(file_path)
