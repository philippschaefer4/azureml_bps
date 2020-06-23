
from azureml.core import Run
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import datetime
import os

run = Run.get_context()

parser = ArgumentParser()
parser.add_argument('--output', dest='prepared_data')
args = parser.parse_args()

# load datasets
df_symptoms = run.input_datasets['symptomcodes'].to_pandas_dataframe()
df = run.input_datasets['df_raw'].to_pandas_dataframe()

###########################################################

# get only data from last t years
t = 5
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

# remove rows with no symptoms
df = df[df['Symptoms']!=()]

##############################################################################

# merge into one row per case
df = df.groupby('Job Card.JobCard Number').apply(lambda x: pd.Series({
    'ProductId': ' '.join(x['Installed Base.InstalledBase ProductID'].unique()),
    'ProductNr': ' '.join(x['Product.Product Number'].unique()),
    'Symptoms': ' '.join(map(str, list(set(x['Symptoms'].sum())))),
    'Start': x['Job Card.Date Start Work'].min(),
    'End': x['Job Card.Date End Work'].max()
  })).reset_index()

###############################################################################

# compute duration column
df = pd.concat([df, pd.DataFrame((df['End'] - df['Start']), columns=['duration'])],axis=1)
df['duration'] = df['duration'].apply(lambda x: x.seconds / 3600)

##############################################################################

# save train and test data
path = arg.prepared_data if args.prepared_data else './outputs'
os.makedirs(path, exist_ok=True)
df.to_csv(path + '/prepared_data.csv', sep=';', header=True, index=False)

run.complete()
