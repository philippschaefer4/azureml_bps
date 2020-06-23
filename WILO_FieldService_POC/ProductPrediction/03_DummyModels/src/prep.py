
from azureml.core import Run
import pandas as pd
import datetime
from pipe import create_pipelines
import os
import numpy as np
import joblib
from argparse import ArgumentParser

t = 0.5
t_test = 0.1

run = Run.get_context()

parser = ArgumentParser()
parser.add_argument('--output', dest='prepared_data')
parser.add_argument('--pipeline_data', dest='pipeline_data')
args = parser.parse_args()

# load datasets
df_symptoms = run.input_datasets['symptomcodes'].to_pandas_dataframe()
df = run.input_datasets['df'].to_pandas_dataframe()

###########################################################

# get only data from last t years
df = df[df['Job Card.Date Start Work']>(datetime.datetime.today() - datetime.timedelta(days=t*365))]

############################################################

# clean data
df = df.replace(['', '0', '-', '000','N/A'], np.nan)
df = df.dropna().reset_index(drop=True)

#############################################################################

# combine Component/Failure Code in train data
df = pd.concat([df, pd.DataFrame(df.apply(lambda x: (x['Job Card.ComponentCode'],x['Job Card.FailureCode']), axis=1), columns=['CompFail'])], axis=1)

# combine Component/Failure Code in symptom table
df_symptoms = df_symptoms[['ComponentCode', 'FailureCode', 'Symptom1', 'Symptom2', 'Symptom3', 'Symptom4']]
df_symptoms = pd.concat([df_symptoms, pd.DataFrame(df_symptoms.apply(lambda x: (x['ComponentCode'],x['FailureCode']),axis=1), columns=['CompFail'])],axis=1)

# merge train data on symptoms
df = pd.merge(df, df_symptoms, on='CompFail', how='left')
df = pd.concat([df, pd.DataFrame(df[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4']].apply(lambda x: tuple([ x[col] for col in ['Symptom1','Symptom2','Symptom3','Symptom4'] if str(x[col]) != 'None' ]), axis=1), columns=['Symptoms'])], axis=1)

# merge into one row per case
df = df.groupby('Job Card.JobCard Number').apply(lambda x: pd.Series({
    'ProductNr': ' '.join(x['Product.Product Number'].unique()),
    'Symptoms': ' '.join(map(str, list(set(x['Symptoms'].sum())))),
    'Start': x['Job Card.Date Start Work'].min(),
    'End': x['Job Card.Date End Work'].max()
  })).reset_index()

df = pd.concat([df, pd.DataFrame((df['End'] - df['Start']), columns=['duration'])],axis=1)
df['duration'] = df['duration'].apply(lambda x: x.seconds / 3600)

##############################################################################

# split data (test data from last t_test years)
df_train = df[df['Start']<(datetime.datetime.today() - datetime.timedelta(days=t_test*365))]
df_test = df[df['Start']>=(datetime.datetime.today() - datetime.timedelta(days=t_test*365))]

##############################################################################

# select columns for training
cfg = {}
cfg['multi_cols'] = ['Symptoms']
cfg['num_target_cols'] = ['duration']
cfg['multi_target_cols'] = ['ProductNr']

feature_pipe, target_pipe = create_pipelines(cfg)
pipelines = { 'feature_pipe': feature_pipe, 'target_pipe': target_pipe }

##############################################################################

X_train = pipelines['feature_pipe'].fit_transform(df_train)
y_train = pipelines['target_pipe'].fit_transform(df_train)
X_test = pipelines['feature_pipe'].transform(df_test)
y_test = pipelines['target_pipe'].transform(df_test)

# rename columns
feature_columns = [ 'feat_'+ str(i) for i in range(X_train.shape[1])]
target_columns = [ 'target_'+ str(i) for i in range(y_train.shape[1])]

df_train = pd.concat([
    pd.DataFrame(X_train, columns=feature_columns),
    pd.DataFrame(y_train, columns=target_columns)
], axis=1)

df_test = pd.concat([
    pd.DataFrame(X_test, columns=feature_columns),
    pd.DataFrame(y_test, columns=target_columns)
], axis=1)

##############################################################################

# save train and test data to run output
os.makedirs('outputs', exist_ok=True)
df_train.to_csv('./outputs/train_data.csv', sep=';', header=True, index=False)
df_test.to_csv('./outputs/test_data.csv', sep=';', header=True, index=False)

# and save train and test data to PipelineData output
os.makedirs(args.prepared_data, exist_ok=True)
df_train.to_csv(args.prepared_data + '/train_data.csv', sep=';', header=True, index=False)
df_test.to_csv(args.prepared_data + '/test_data.csv', sep=';', header=True, index=False)

# save pipelines only in run output
joblib.dump(pipelines, './outputs/pipelines.pkl')

# and save in PipelineData output
joblib.dump(pipelines, args.pipeline_data)# + '/pipelines.pkl')

run.complete()
