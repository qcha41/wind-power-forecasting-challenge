"""
Example of custom metric script.
The custom metric script must contain the definition of custom_metric_function and a main function
that reads the two csv files with pandas and evaluate the custom metric.
"""

# TODO: add here the import necessary to your metric function
import numpy as np

def CAPE_CNR_function(dataframe_y_true, dataframe_y_pred):
    """
CAPE (Cumulated Absolute Percentage Error) function used by CNR for the evaluation of predictions

    Args
        dataframe_y_true: Pandas Dataframe
            Dataframe containing the true values of y.
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_true = pd.read_csv(CSV_1_FILE_PATH, index_col=0, sep=',')

        dataframe_y_pred: Pandas Dataframe
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_pred = pd.read_csv(CSV_2_FILE_PATH, index_col=0, sep=',')

    Returns
        score: Float
            The metric evaluated with the two dataframes. This must not be NaN.
    """

    # CAPE function
    cape_cnr = 100*np.sum(np.abs(dataframe_y_pred-dataframe_y_true))/np.sum(dataframe_y_true)
    return cape_cnr


if __name__ == '__main__':
    import pandas as pd
    CSV_FILE_Y_TRUE = 'Y_test.csv'
    CSV_FILE_Y_PRED = 'Y_test_benchmark.csv'
    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
    df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')
    print(CAPE_CNR_function(df_y_true, df_y_pred))
