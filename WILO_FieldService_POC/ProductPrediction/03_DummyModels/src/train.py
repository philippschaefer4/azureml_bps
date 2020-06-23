
from azureml.core import Run
import os
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.dummy import DummyRegressor
from sklearn.metrics import recall_score, precision_score, hamming_loss, zero_one_loss, mean_absolute_error, mean_squared_error, r2_score

run = Run.get_context()

# load data
train_data = run.input_datasets['train_data'].to_pandas_dataframe()
test_data = run.input_datasets['test_data'].to_pandas_dataframe()

# split train/test and feat/target
X_train = train_data[[ col for col in train_data.columns if col.startswith('feat')]]
y_train = train_data[[ col for col in train_data.columns if col.startswith('target')]].drop(['target_0'], axis=1)
X_test = test_data[[col for col in test_data.columns if col.startswith('feat')]]
y_test = test_data[[ col for col in test_data.columns if col.startswith('target')]].drop(['target_0'], axis=1)

############################################################

# train classifier
model = MultiOutputClassifier(DummyClassifier(strategy='stratified'))
model.fit(X_train, y_train)

# evaluate test data
y_pred = model.predict(X_test)
run.log('precision_macro', precision_score(y_test, y_pred, average='macro'))
run.log('precision_samples', precision_score(y_test, y_pred, average='samples'))
run.log('recall_macro', recall_score(y_test, y_pred, average='macro'))
run.log('recall_samples', recall_score(y_test, y_pred, average='samples'))
run.log('hamming_loss', hamming_loss(y_test, y_pred))
run.log('zero_one_loss', zero_one_loss(y_test, y_pred))

# evaluate train data
y_pred = model.predict(X_train)
run.log('precision_macro_train', precision_score(y_train, y_pred, average='macro'))
run.log('precision_samples_train', precision_score(y_train, y_pred, average='samples'))
run.log('recall_macro_train', recall_score(y_train, y_pred, average='macro'))
run.log('recall_samples_train', recall_score(y_train, y_pred, average='samples'))
run.log('hamming_loss_train', hamming_loss(y_train, y_pred))
run.log('zero_one_loss_train', zero_one_loss(y_train, y_pred))

# save model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

############################################################

# train regressor
X_train = train_data[[ col for col in train_data.columns if col.startswith('feat')]]
y_train = train_data[[ col for col in train_data.columns if col.startswith('target')]][['target_0']]
X_test = test_data[[col for col in test_data.columns if col.startswith('feat')]]
y_test = test_data[[ col for col in test_data.columns if col.startswith('target')]][['target_0']]

model_regressor = DummyRegressor(strategy="mean")
model_regressor.fit(X_train, y_train)

y_pred = model_regressor.predict(X_test)
run.log('mae', mean_absolute_error(y_test, y_pred))
run.log('mse', mean_squared_error(y_test, y_pred))
run.log('r2', r2_score(y_test, y_pred))

y_pred = model_regressor.predict(X_train)
run.log('mae_train', mean_absolute_error(y_train, y_pred))
run.log('mse_train', mean_squared_error(y_train, y_pred))
run.log('r2_train', r2_score(y_train, y_pred))

# save regressor model
joblib.dump(value=model_regressor, filename='outputs/model_regressor.pkl')

run.complete()
