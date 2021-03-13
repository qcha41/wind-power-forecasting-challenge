import os
import mlflow
import numpy as np
import pandas as pd

from .utilities import get_nwp_cols, TEMP_FOLDER
from . import plots


class ModelWrapper:
    def __init__(self):
        self.model = None
        self.history = None

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        """ Starts the procedure of training (train + plot learning curves """

        self.run(x_train, y_train, x_valid, y_valid)
        plots.learning_curves(self.history)

    def predict(self, t, x, y_true, label):
        """ Use the trained model to predict on provided data x and compare with y_true """

        y_predict = pd.Series(self.model.predict(x).squeeze(), index=t.index, name="Production")
        loss = np.mean((y_true - y_predict) ** 2)
        plots.predictions_vs_time(t, y_true, y_predict, label, loss)
        return y_predict, loss


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
    file_path = os.path.join(TEMP_FOLDER, 'predictions.csv')
    predictions.to_csv(file_path)
    mlflow.log_artifact(file_path)
