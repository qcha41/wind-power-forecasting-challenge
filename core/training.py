import os
import mlflow
import numpy as np
import pandas as pd

from .utilities import get_nwp_cols, TEMP_FOLDER


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
    """ Save given dataframe into a csv file and upload it in mlflow"""

    file_path = os.path.join(TEMP_FOLDER, filename)
    data.to_csv(file_path)
    mlflow.log_artifact(file_path)


