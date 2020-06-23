
from azureml.core import Run
import os
import joblib
from argparse import ArgumentParser
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import recall_score, precision_score, hamming_loss, zero_one_loss, mean_absolute_error, mean_squared_error, r2_score

run = Run.get_context()

parser = ArgumentParser()
parser.add_argument('--input', dest='preprocessed_data')
args = parser.parse_args()

# ############################################################

# # load data
# if args.preprocessed_data:
#     train_data = pd.read_csv(args.preprocessed_data + '/train_data.csv', sep=';', header=0)
#     test_data = pd.read_csv(args.preprocessed_data + '/test_data.csv', sep=';', header=0)
# else:
#     train_data = run.input_datasets['train_data'].to_pandas_dataframe()
#     test_data = run.input_datasets['test_data'].to_pandas_dataframe()
    
# ############################################################

# # split train/test and feat/target for classifier
# X_train = train_data[[ col for col in train_data.columns if col.startswith('feat')]]
# y_train = train_data[[ col for col in train_data.columns if col.startswith('target')]].drop(['target_0'], axis=1)
# X_test = test_data[[col for col in test_data.columns if col.startswith('feat')]]
# y_test = test_data[[ col for col in test_data.columns if col.startswith('target')]].drop(['target_0'], axis=1)

# ############################################################

# # train classifier
# model = MultiOutputClassifier(SGDClassifier())
# model.fit(X_train, y_train)

# ############################################################

# # evaluate test data
# y_pred = model.predict(X_test).round()
# run.log_table(
#     'test_evaluation_classification',
#     {
#         'precision_macro': [precision_score(y_test, y_pred, average='macro')],
#         'precision_samples': [precision_score(y_test, y_pred, average='samples')],
#         'recall_macro': [recall_score(y_test, y_pred, average='macro')],
#         'recall_samples': [recall_score(y_test, y_pred, average='samples')],
#         'hamming_loss': [hamming_loss(y_test, y_pred)],
#         'zero_one_loss': [zero_one_loss(y_test, y_pred)]
#     }
# )

# # evaluate train data
# y_pred = model.predict(X_train).round()
# run.log_table(
#     'train_evaluation_classification',
#     {
#         'precision_macro_train': [precision_score(y_train, y_pred, average='macro')],
#         'precision_samples_train': [precision_score(y_train, y_pred, average='samples')],
#         'recall_macro_train': [recall_score(y_train, y_pred, average='macro')],
#         'recall_samples_train': [recall_score(y_train, y_pred, average='samples')],
#         'hamming_loss_train': [hamming_loss(y_train, y_pred)],
#         'zero_one_loss_train': [zero_one_loss(y_train, y_pred)]
#     }
# )

# ############################################################

# # save model
# os.makedirs('outputs', exist_ok=True)
# joblib.dump(value=model, filename='outputs/model.pkl')

# ############################################################

# # split train/test and feat/target for regressor
# X_train = train_data[[ col for col in train_data.columns if col.startswith('feat')]]
# y_train = train_data[[ col for col in train_data.columns if col.startswith('target')]][['target_0']]
# X_test = test_data[[col for col in test_data.columns if col.startswith('feat')]]
# y_test = test_data[[ col for col in test_data.columns if col.startswith('target')]][['target_0']]
# y_test = y_test.values[:,0]

# ############################################################

# # train regressor
# model_regressor = SGDRegressor()
# model_regressor.fit(X_train, y_train)

# ############################################################

# # evaluate test data
# y_pred = model_regressor.predict(X_test)
# run.log_table(
#     'test_evaluation_regression',
#     {
#         'mae': [mean_absolute_error(y_test, y_pred)],
#         'mse': [mean_squared_error(y_test, y_pred)],
#         'r2': [r2_score(y_test, y_pred)]
#     }
# )

# # evaluate train data
# y_pred = model_regressor.predict(X_train)
# run.log_table(
#     'train_evaluation_regression',
#     {
#         'mae_train': [mean_absolute_error(y_train, y_pred)],
#         'mse_train': [mean_squared_error(y_train, y_pred)],
#         'r2_train': [r2_score(y_train, y_train)]
#     }
# )

# ############################################################

# # save regressor model
# joblib.dump(value=model_regressor, filename='outputs/model_regressor.pkl')


############################################################

# load data
if args.preprocessed_data:
    train_data = joblib.load(args.preprocessed_data + '/train_data.pkl')
    test_data = joblib.load(args.preprocessed_data + '/test_data.pkl')
else:
    train_data = joblib.load(run.input_datasets['train_data'] + '/workspaceblobstore/SKLearnPrediction' + '/train_data.pkl')
    test_data = joblib.load(run.input_datasets['test_data'] + '/workspaceblobstore/SKLearnPrediction' + '/test_data.pkl')
    
print(train_data.keys())
print(test_data.keys())

run.complete()
