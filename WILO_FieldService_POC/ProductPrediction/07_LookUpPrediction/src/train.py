
from azureml.core import Run

import os
import joblib
from argparse import ArgumentParser
from lookup import LookUpClassifier
from sklearn.metrics import recall_score, precision_score, hamming_loss, zero_one_loss, mean_absolute_error, mean_squared_error, r2_score

run = Run.get_context()

parser = ArgumentParser()
parser.add_argument('--input', dest='prepared_data')
args = parser.parse_args()

############################################################

print('\n#####################################################')
print('loaded')
print('\n#####################################################')

# load data
if args.prepared_data:
    train_data = pd.read_csv(args.prepared_data + '/train_data.csv', sep=';', header=0)
    test_data = pd.read_csv(args.prepared_data + '/test_data.csv', sep=';', header=0)
else:
    train_data = run.input_datasets['train_data'].to_pandas_dataframe()
    test_data = run.input_datasets['test_data'].to_pandas_dataframe()
    
train_data = train_data.dropna().reset_index(drop=True)
test_data = test_data.dropna().reset_index(drop=True)
    
#################################################################

print('\n#####################################################')
print('train')
print('\n#####################################################')

# train classifier
model = LookUpClassifier(threshold=0.2)
model.fit(train_data)

print('\n#####################################################')
print('trained')
print('\n#####################################################')

############################################################

X_test = test_data[['ProductId', 'Country', 'Symptoms']].values.tolist()
y_test = test_data['ProductNrs'].values.tolist() 

X_train = train_data[['ProductId', 'Country', 'Symptoms']].values.tolist()
y_train = train_data['ProductNrs'].values.tolist() 

############################################################

# # evaluate test data
# y_pred = model.predict(X_test)
# y_pred_tr = model.transform_products(y_pred)
# y_test_tr = model.transform_products(y_test)
# run.log_table(
#     'test_evaluation_classification',
#     {
#         'precision_macro': [precision_score(y_test_tr, y_pred_tr, average='macro')],
#         'precision_samples': [precision_score(y_test_tr, y_pred_tr, average='samples')],
#         'recall_macro': [recall_score(y_test_tr, y_pred_tr, average='macro')],
#         'recall_samples': [recall_score(y_test_tr, y_pred_tr, average='samples')],
#         'hamming_loss': [hamming_loss(y_test_tr, y_pred_tr)],
#         'zero_one_loss': [zero_one_loss(y_test_tr, y_pred_tr)]
#     }
# )

# # evaluate train data
# y_pred = model.predict(X_train)
# y_pred_tr = model.transform_products(y_pred)
# y_train_tr = model.transform_products(y_train)
# run.log_table(
#     'train_evaluation_classification',
#     {
#         'precision_macro_train': [precision_score(y_train_tr, y_pred_tr, average='macro')],
#         'precision_samples_train': [precision_score(y_train_tr, y_pred_tr, average='samples')],
#         'recall_macro_train': [recall_score(y_train_tr, y_pred_tr, average='macro')],
#         'recall_samples_train': [recall_score(y_train_tr, y_pred_tr, average='samples')],
#         'hamming_loss_train': [hamming_loss(y_train_tr, y_pred_tr)],
#         'zero_one_loss_train': [zero_one_loss(y_train_tr, y_pred_tr)]
#     }
# )

############################################################

# save model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()
