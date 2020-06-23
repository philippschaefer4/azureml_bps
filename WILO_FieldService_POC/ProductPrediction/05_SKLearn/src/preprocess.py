
from azureml.core import Run

import datetime
import os
import pandas as pd
import joblib
from argparse import ArgumentParser

from pipe import create_pipelines

run = Run.get_context()

parser = ArgumentParser()
parser.add_argument('--input', dest='prepared_data')
parser.add_argument('--output', dest='preprocessed_data')
args = parser.parse_args()

# load datasets
if args.prepared_data:
    df = pd.read_csv(args.prepared_data + '/prepared_data.csv', sep=';', header=0)
else:
    df = run.input_datasets['df_prepared'].to_pandas_dataframe()
    
print('\n############################################################################################\n')
print(df.columns)
print('\n############################################################################################\n')

##############################################################################

# split data (test data from last t_test years)
t_test = 0.5
df_train = df[df['Start']<(datetime.datetime.today() - datetime.timedelta(days=t_test*365))]
df_test = df[df['Start']>=(datetime.datetime.today() - datetime.timedelta(days=t_test*365))]

##############################################################################

# select columns for training
cfg = {}
cfg['multi_cols'] = ['Symptoms']
cfg['num_target_cols'] = ['duration']
cfg['multi_target_cols'] = ['ProductNr']

# create pipeline
pipelines = create_pipelines(cfg)

# fit pipelines and transform data
X_train = pipelines['feature_pipe'].fit_transform(df_train)
y_train = pipelines['target_pipe'].fit_transform(df_train)
X_test = pipelines['feature_pipe'].transform(df_test)
y_test = pipelines['target_pipe'].transform(df_test)

##############################################################################

# rename columns
feature_columns = [ 'feat_'+ str(i) for i in range(X_train.shape[1])]
target_columns = [ 'target_'+ str(i) for i in range(y_train.shape[1])]

df_train = pd.concat([
    pd.DataFrame(X_train, columns=feature_columns),
    pd.DataFrame(y_train, columns=target_columns)
], axis=1)

print('\n############################################################################################\n')
print(df_train.columns)
print('\n############################################################################################\n')

df_test = pd.concat([
    pd.DataFrame(X_test, columns=feature_columns),
    pd.DataFrame(y_test, columns=target_columns)
], axis=1)

##############################################################################

# save train and test data
path = args.preprocessed_data if args.preprocessed_data else './outputs'
os.makedirs(path, exist_ok=True)
df_train.to_csv(path + '/train_data.csv', sep=';', header=True, index=False)
df_test.to_csv(path + '/test_data.csv', sep=';', header=True, index=False)

# save pipelines
os.makedirs('outputs', exist_ok=True)
joblib.dump(pipelines, './outputs/pipelines.pkl')

###########################################################

run.complete()
