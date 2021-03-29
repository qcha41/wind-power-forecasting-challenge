import tempfile
import os
import pandas as pd
import mlflow

TEMP_FOLDER = tempfile.mkdtemp()
mlflow.tensorflow.autolog(every_n_iter=1,log_models=False)

def get_nwp_cols(df):
    """ Returns the list of NWP columns in the dataframe """

    return [col for col in df.columns if col.startswith('NWP')]


def configure_mlflow_server():
    """ Configure tracking uri of mlflow server """

    try:
        from google.colab import drive
        drive.mount('/content/gdrive')
        cred_path = '/content/gdrive/MyDrive/wind_power_forecasting_challenge_aws_credentials.csv'
    except:
        cred_path = './local/aws_credentials.csv'
    cred = pd.read_csv(cred_path, index_col=0, squeeze=True)
    os.environ['AWS_ACCESS_KEY_ID'] = cred.AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = cred.AWS_SECRET_ACCESS_KEY
    mlflow.set_tracking_uri(f"http://{cred.AWS_URL}")
