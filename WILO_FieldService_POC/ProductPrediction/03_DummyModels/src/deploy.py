
from azureml.core import Run, Model
import os
import pandas as pd
import joblib
from argparse import ArgumentParser

run = Run.get_context()
ws = run.experiment.workspace

parser = ArgumentParser()
parser.add_argument('--pipeline_data', dest='pipeline_data')
parser.add_argument('--trained_classifier', dest='trained_classifier')
parser.add_argument('--trained_regressor', dest='trained_regressor')
args = parser.parse_args()

# Model.register(args.pipeline_data, 'DummyPipe', ws)
# Model.register(args.trained_classifier, 'DummyModel', ws)
# Model.register(args.trained_regressor, 'DummyModelRegressor', ws)

for child in run.parent.get_children():
    if child.name == 'prep.py':
        child.register_model('DummyPipe', 'outputs/pipelines.pkl')
    elif child.name == 'train.py':
        child.register_model('DummyModel', 'outputs/model.pkl')
        child.register_model('DummyModelRegressor', 'outputs/model_regressor.pkl')
        
run.complete()
