import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(11.7,8.27)})

import utilities

def unwrap_data(data):
    subdata_list = []
    for col_name in [name for name in data.columns if name.startswith('NWP')]:
        subdata = data[[col for col in data.columns if col==col_name or 'NWP' not in col]]
        subdata.loc[:,'NWP_num'] = int(col_name.split('_')[0].strip('NWP'))
        subdata.loc[:,'NWP_var'] = col_name.split('_')[1]
        subdata.rename(columns={col_name:'value'}, inplace=True)
        subdata.reset_index(inplace=True)
        subdata_list.append(subdata)
    new_data = pd.concat(subdata_list, ignore_index=True)
    return new_data

def plot_feature_in_time(data,feature):
    g = sns.FacetGrid(data, col="WF", col_wrap=3)
    g.map(sns.lineplot, "Time", feature)
    for axes in g.axes.flat:
        axes.set_xticklabels(axes.get_xticklabels(), rotation=30)
    plt.savefig(f'./plots/{feature}_vs_time.jpg',dpi=600)

def plot_feature_in_time_allmean(data):
    for col in [col for col in data.columns if col == "Production" or col.startswith('NWP0')]:
        plot_feature_in_time(data, col)
        plt.close()

def plot_feature_influence(data_unwrapped, feature):
    subdata = data_unwrapped[data_unwrapped.NWP_var==feature].rename(columns={'value':feature})
    g = sns.FacetGrid(subdata, col="WF", col_wrap=3, hue='NWP_num')
    g.map(sns.scatterplot, feature, "Production", alpha=0.5)
    g.add_legend()
    g.fig.subplots_adjust(top=0.9)
    plt.savefig(f'./plots/production_vs_{feature}.jpg',dpi=600)

def plot_feature_influence_all(data_unwrapped):
    for feature in [f for f in data_unwrapped.NWP_var.unique()] :
        plot_feature_influence(data_unwrapped,feature)
        plt.close()

if __name__ == '__main__' :

    forecast_memory = 0.9

    # Load data
    df = utilities.load_data()

    # Preprocess data
    df = utilities.calculate_best_forecasts(df, forecast_memory)
    df = utilities.interpolate_nans(df)
    df = utilities.augment_data(df)

    df_unwrapped = unwrap_data(df)

    plot_feature_influence_all(df_unwrapped)
    plot_feature_in_time(df,"Production")
    plot_feature_in_time_allmean(df)

    plot_feature_in_time(df, "NWP0_WS")
    plt.show()



