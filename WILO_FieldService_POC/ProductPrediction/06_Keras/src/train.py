
from azureml.core import Run

import os
import joblib
import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import recall_score, precision_score, hamming_loss, zero_one_loss, mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense

run = Run.get_context()

parser = ArgumentParser()
parser.add_argument('--input', dest='preprocessed_data')
args = parser.parse_args()

############################################################

# load data
if args.preprocessed_data:
    train_data = pd.read_csv(args.preprocessed_data + '/train_data.csv', sep=';', header=0)
    test_data = pd.read_csv(args.preprocessed_data + '/test_data.csv', sep=';', header=0)
else:
    train_data = run.input_datasets['train_data'].to_pandas_dataframe()
    test_data = run.input_datasets['test_data'].to_pandas_dataframe()
    
############################################################

# split train/test and feat/target
X_train = train_data[[ col for col in train_data.columns if col.startswith('feat')]].values
y_train = train_data[[ col for col in train_data.columns if col.startswith('target')]].drop(['target_0'], axis=1).values
X_test = test_data[[col for col in test_data.columns if col.startswith('feat')]].values
y_test = test_data[[ col for col in test_data.columns if col.startswith('target')]].drop(['target_0'], axis=1).values

############################################################

# create model
inputs = Input(shape=(X_train.shape[1],))

x = Dense(512, activation='relu')(inputs)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)

outputs = Dense(y_train.shape[1], activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
run.log('Model Summary', model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
# tf.keras.metrics.Recall(top_k=y_train.shape[1])

# train classifier
history = model.fit(X_train, y_train, batch_size=64, epochs=10)

##############################################################

# evaluate test data
y_pred = model.predict(X_test).round()
run.log_table(
    'test_evaluation_classification',
    {
        'precision_macro': [precision_score(y_test, y_pred, average='macro')],
        'precision_samples': [precision_score(y_test, y_pred, average='samples')],
        'recall_macro': [recall_score(y_test, y_pred, average='macro')],
        'recall_samples': [recall_score(y_test, y_pred, average='samples')],
        'hamming_loss': [hamming_loss(y_test, y_pred)],
        'zero_one_loss': [zero_one_loss(y_test, y_pred)]
    }
)

# evaluate train data
y_pred = model.predict(X_train).round()
run.log_table(
    'train_evaluation_classification',
    {
        'precision_macro_train': [precision_score(y_train, y_pred, average='macro')],
        'precision_samples_train': [precision_score(y_train, y_pred, average='samples')],
        'recall_macro_train': [recall_score(y_train, y_pred, average='macro')],
        'recall_samples_train': [recall_score(y_train, y_pred, average='samples')],
        'hamming_loss_train': [hamming_loss(y_train, y_pred)],
        'zero_one_loss_train': [zero_one_loss(y_train, y_pred)]
    }
)

# save model
os.makedirs('outputs', exist_ok=True)
#joblib.dump(value=model, filename='outputs/model.pkl')
model.save('outputs/model.pkl')

############################################################

# train regressor
X_train = train_data[[ col for col in train_data.columns if col.startswith('feat')]].values
y_train = train_data[[ col for col in train_data.columns if col.startswith('target')]][['target_0']].values
X_test = test_data[[col for col in test_data.columns if col.startswith('feat')]].values
y_test = test_data[[ col for col in test_data.columns if col.startswith('target')]][['target_0']].values
y_test = y_test.values[:,0]

############################################################

# create model
inputs = Input(shape=(X_train.shape[1],))

x = Dense(512, activation='relu')(inputs)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)


outputs = Dense(y_train.shape[1], activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
run.log('Model Summary', model.summary())

model.compile(loss='root_mean_squared_error', optimizer='adam', metrics=["mae"])
# tf.keras.metrics.Recall(top_k=y_train.shape[1])

# train classifier
history = model.fit(X_train, y_train, batch_size=64, epochs=10)

############################################################

# evaluate test data
y_pred = model_regressor.predict(X_test)
run.log_table(
    'test_evaluation_regression',
    {
        'mae': [mean_absolute_error(y_test, y_pred)],
        'mse': [mean_squared_error(y_test, y_pred)],
        'r2': [r2_score(y_test, y_pred)]
    }
)

# evaluate train data
y_pred = model_regressor.predict(X_train)
run.log_table(
    'train_evaluation_regression',
    {
        'mae_train': [mean_absolute_error(y_train, y_pred)],
        'mse_train': [mean_squared_error(y_train, y_pred)],
        'r2_train': [r2_score(y_train, y_train)]
    }
)

############################################################

# save regressor model
# joblib.dump(value=model_regressor, filename='outputs/model_regressor.pkl')
model.save('outputs/model_regressor.pkl')

run.complete()
