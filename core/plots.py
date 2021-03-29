import os
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpld3

from .utilities import TEMP_FOLDER

sns.set_theme(style="darkgrid")
sns.plotting_context('notebook', rc={'xtick.labelsize': 12})


def feature_vs_time(df, nwp_var, title=None, type='lineplot', hue="NWP"):
    """ Plots a given feature as a function of time, grouped by nwp models """

    nwp_runs = [col for col in df.columns if col.startswith('NWP') and col.split('_')[1] == nwp_var]
    df = pd.concat([separate_feature_info(df, run) for run in nwp_runs]).sort_values(by='Time')
    g = sns.FacetGrid(df, col="WF", hue=hue, col_wrap=3)
    g.map(getattr(sns, type), "Time", nwp_var, alpha=0.7)
    g.add_legend()
    rotate_x_labels(g)
    if title is not None:
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(title)


def production_vs_feature(df, nwp_var):
    """ Plots the production data as a function of the given feature, grouped by nwp models """

    nwp_runs = [col for col in df.columns if col.startswith('NWP') and col.split('_')[1] == nwp_var]
    df = pd.concat([separate_feature_info(df, run, with_prod=True) for run in nwp_runs]).sort_values(by='Time')
    g = sns.FacetGrid(df, col="WF", hue="NWP", col_wrap=3)
    g.map(sns.scatterplot, nwp_var, "Production", alpha=0.5)
    g.add_legend()
    rotate_x_labels(g)


def production_vs_time(df):
    """ Plots the production data as a function of time """
    g = sns.FacetGrid(df, col="WF", col_wrap=3)
    g.map(sns.lineplot, "Time", "Production")
    rotate_x_labels(g)


def correlation_graph(df):
    """ Plots the correlation matrix of the dataset """

    def unit_wf_corr_graph(data, color):
        """ Plot the correlation matrix of a given wind farm """

        corr = data.drop(columns="WF").corr()
        mask = np.zeros(corr.shape, dtype=bool)
        mask[np.triu_indices(len(mask))] = True
        a = sns.heatmap(corr, cmap="coolwarm", mask=mask, vmin=-1, vmax=1, cbar_ax=cbar_ax, annot=True)
        a.set_ylim(len(corr), 1)
        a.set_xlim(0, len(corr.columns) - 1)

    g = sns.FacetGrid(df, col="WF", col_wrap=3)
    cbar_ax = g.fig.add_axes([1.0, .2, .02, .7])
    g.map_dataframe(unit_wf_corr_graph)
    rotate_x_labels(g)


def rotate_x_labels(graph):
    """ Rotate the x labels of the given graph """

    for axes in graph.axes.flat:
        plt.setp(axes.xaxis.get_majorticklabels(), rotation=60)


def separate_feature_info(df, run, with_prod=False):
    """ Separate run data (Time, WF, run_value) into (Time, WF, nwp_num, run_value) """

    nwp_num = int(run.split('_')[0].strip('NWP'))
    nwp_var = run.split('_')[1]
    base_columns = [col for col in df.columns if "NWP" not in col and col != "Production"] + [run]
    if with_prod is True:
        base_columns.append("Production")
    df_separated = df[base_columns] \
        .assign(NWP=nwp_num) \
        .rename(columns={run: nwp_var})
    return df_separated


def learning_curves(history):
    """ Plots the training and validation error as a function of the number of epochs """

    plt.figure()
    for key in history.history.keys():
        label = f'{key}\nfinal={(history.history[key][-1]):.2f}'
        ax = sns.lineplot(x=range(len(history.history[key])), y=history.history[key], label=label)
    ax.set_xlabel('epochs')
    save_plot(plt.gcf(), 'learning_curves')


def predictions_vs_time(time, y_true, y_predict, title):
    """ Plots true (and predicted if provided) production data as function of time """

    plt.figure(figsize=(15, 6))
    sns.lineplot(x=time, y=y_predict, label='y_predict')
    if y_true is not None:
        sns.lineplot(x=time, y=y_true, label='y_true')
    plt.title(title)
    save_plot(plt.gcf(), title)


def inactivity_histogram(inactivity_periods, delay_threshold):
    """ Plots the histogram of the periods of inactivity (consecutive time of null production)"""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    kept = [p['hours'] for p in inactivity_periods if p['hours'] <= delay_threshold]
    ignored = [p['hours'] for p in inactivity_periods if p['hours'] > delay_threshold]
    sns.histplot(x=kept, binwidth=3, ax=ax, label=f'kept ({sum(kept) + len(kept)} examples)')
    sns.histplot(x=ignored, binwidth=3, ax=ax, label=f'ignored ({sum(ignored) + len(ignored)} examples)', color='red')
    ax.plot([24, 24], ax.get_ylim(), 'r', label='24h threshold')
    ax.set_yscale('log')
    ax.set_xlabel("Period of inactivity [hours]")
    ax.set_title("Periods of inactivity over all wind farms")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)


def save_plot(fig, file_name):
    """ Save given figure """

    file_path = os.path.join(TEMP_FOLDER, file_name + '.html')
    mpld3.save_html(fig, file_path)
    mlflow.log_artifact(file_path)
