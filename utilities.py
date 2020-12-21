import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()
import datetime as dt
import tensorflow as tf
import CAPE_CNR_metric
import numpy as np
import model

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
        self.data = pd.read_csv('data/data_processed.csv', index_col=0)
        self.data.loc[:, 'Time'] = pd.to_datetime(self.data['Time'])

        # Training models (one for each WF)
        self.models = {}
        for WF_num in self.data.WF.unique() :
            self.models[WF_num] = ModelRNN(self, WF_num)


    def process_wf(self, WF_num, **kwargs):
        self.models[WF_num].process(**kwargs)

    def process_all_wf(self):
        for WF_num in self.models.keys():
            self.process_wf(WF_num)
        self.output_predictions()

    def output_predictions(self):
        predictions = pd.concat([self.models[i].Y_predict['test'] for i in self.data.WF.unique()])
        predictions.to_csv(os.path.join(self.folder_path, 'Y_test.csv'), float_format='%.2f')


class ModelTrainer:

    def __init__(self, manager, WF_num):

        self.manager = manager
        self.WF_num = WF_num
        self.data = self.manager.data.loc[self.manager.data.WF == self.WF_num].drop(columns='WF')

        self.folder_path = os.path.join(self.manager.folder_path, f'model_{WF_num}')
        os.mkdir(self.folder_path)

        self.window_size = 96  # In hours
        self.shuffle_buffer = 10000
        self.batch_size = 1000
        self.split_train = 0.85

        self.X = {}
        self.Y = {}
        self.Y_predict = {}

        self.scores = {}
        self.history = []

    def prepare_data(self):

        NWP_names = [col for col in self.data.columns if col.startswith('NWP')]

        global_train_length = len(self.data.Production.dropna())
        nb_train_examples = int((global_train_length - self.window_size) * self.split_train)
        self.X['train'] = self.data.iloc[:nb_train_examples][NWP_names]
        self.Y['train'] = self.data.iloc[self.window_size - 1:nb_train_examples]['Production']
        self.X['valid'] = self.data.iloc[nb_train_examples - (self.window_size - 1):global_train_length][NWP_names]
        self.Y['valid'] = self.data.iloc[nb_train_examples:global_train_length]['Production']
        self.X['test'] = self.data.iloc[global_train_length - (self.window_size - 1):][NWP_names]
        self.Y['test'] = self.data.iloc[global_train_length:]['Production']

    def calculate_scores(self):
        for key in ['train', 'valid']:
            score = {'mse': np.mean(np.square(self.Y[key] - self.Y_predict[key])),
                     'cape': CAPE_CNR_metric.CAPE_CNR_function(self.Y[key], self.Y_predict[key])}
            self.scores[key] = score

    def plot_predictions(self):
        fig = plt.figure()
        for key in ['train', 'valid', 'test']:
            fig.clf()
            ax = fig.add_subplot()
            time = self.data.Time.loc[self.Y[key].index]
            ax.plot(time, self.Y[key], label='true')
            if key in ['train','valid']:
                predict_label = f'predict\nMSE={self.scores[key]["mse"]:g}\nCAPE={self.scores[key]["cape"]:g}'
            else:
                predict_label = 'predict'
            ax.plot(time, self.Y_predict[key], '--', label=predict_label)
            ax.legend()
            self.save_plot(fig, f'prediction_{key}')
            current_xlim = ax.get_xlim()
            ax.set_xlim((current_xlim[1] + current_xlim[0])/2,
                        (current_xlim[1] + current_xlim[0])/2 + (current_xlim[1] - current_xlim[0]) * 0.1)
            self.save_plot(fig, f'prediction_{key}_x10')
        plt.close(fig)

    def save_plot(self, fig, title):
        fig.suptitle(title)
        fig.savefig(os.path.join(self.folder_path, f'{title}.jpg'))


class ModelPersistent(ModelTrainer):

    def __init__(self, manager, WF_num):
        ModelTrainer.__init__(self, manager, WF_num)

    def process(self, **kwargs):
        self.predict()
        self.calculate_scores()
        self.plot_predictions()

    def predict(self):
        for key in ['train', 'valid']:
            self.Y_predict[key] = self.Y[key].copy()
            self.Y_predict[key].loc[:] = self.data.loc[self.Y[key].index.values - 1, 'Production'].values
        self.Y_predict['test'] = self.Y['test'].copy()
        self.Y_predict['test'].loc[:] = self.Y['test'].values

class ModelRNN(ModelTrainer):

    def __init__(self, manager, WF_num):
        ModelTrainer.__init__(self, manager, WF_num)

        self.epochs = 500

        self.datasets_train = {}
        self.datasets_predict = {}

        self.model = None

        self.initialize()

    def initialize(self):

        self.prepare_data()
        self.prepare_datasets()
        self.define_model()

    def prepare_datasets(self):
        # Making sequences datasets
        self.datasets_train['train'] = self.windowed_dataset(self.X['train'], production=self.Y['train'])
        self.datasets_train['valid'] = self.windowed_dataset(self.X['valid'], production=self.Y['valid'])
        self.datasets_predict['train'] = self.windowed_dataset(self.X['train'])
        self.datasets_predict['valid'] = self.windowed_dataset(self.X['valid'])
        self.datasets_predict['test'] = self.windowed_dataset(self.X['test'])

    def define_model(self):

        # Define NN model - Content publicly hidden for obvious competitive reasons
        self.model = model.get_compiled_model([self.window_size, self.X['train'].shape[1]])

    def process(self, **kwargs):

        self.train(**kwargs)
        self.predict()
        self.calculate_scores()
        self.plot_learning_curves()
        self.plot_predictions()

    def train(self, epochs=None):
        if epochs is None: epochs = self.epochs

        best_model_weights_filepath = os.path.join(self.folder_path,'best_weights')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_weights_filepath,
                                                                       save_weights_only=True,
                                                                       monitor='val_loss',
                                                                       mode='min',
                                                                       save_best_only=True)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,patience=30)

        # Training
        history = self.model.fit(self.datasets_train['train'], validation_data=self.datasets_train['valid'],
                                 epochs=epochs ,verbose=1,
                                 callbacks=[early_stopping_callback,model_checkpoint_callback])
        self.history.append(history)
        self.model.load_weights(best_model_weights_filepath)

    def predict(self):
        for key in self.datasets_predict.keys():
            self.Y_predict[key] = self.Y[key].copy()
            self.Y_predict[key].loc[:] = self.model.predict(self.datasets_predict[key]).squeeze()

    def plot_learning_curves(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        i = 0
        for h in self.history:
            length = len(h.history['loss'])
            for key in ['loss', 'val_loss']:
                ax.plot(range(i, i + length), h.history[key], label=key)
            i += length
        ax.legend()
        ax.set(xlabel='epoch', ylabel='loss')
        self.save_plot(fig, 'learning_curve')
        ax.set_yscale('log')
        self.save_plot(fig, 'learning_curve_log')
        plt.close(fig)


    def windowed_dataset(self, forecasts, production=None):

        # Forecasts
        dataset = tf.data.Dataset.from_tensor_slices(forecasts)
        dataset = dataset.window(self.window_size, shift=1, drop_remainder=True)  # Make training windows of fixed size
        dataset = dataset.flat_map(lambda w: w.batch(self.window_size, drop_remainder=True))  # Transform windows in batches

        # Production
        if production is not None:
            production_dataset = tf.data.Dataset.from_tensor_slices(production)
            dataset = tf.data.Dataset.zip((dataset, production_dataset))  # Merge forecasts and production
            dataset = dataset.shuffle(self.shuffle_buffer)  # Shuffle the individual batches

        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)  # Make mini-batches of <batch_size> individual batches
        dataset = dataset.prefetch(1)  # Always preload 1 mini-batch in advance to be ready to consume data

        return dataset
