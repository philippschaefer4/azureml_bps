
import json
import numpy as np
import os
from azureml.core.model import Model
import joblib
from pipe import create_pipeline
import pandas as pd

def init():
    global model
    global regressor
    global pipelines
    model_path = Model.get_model_path('DummyModel')
    model = joblib.load(model_path)
    regressor_path = Model.get_model_path('DummyModelRegressor')
    regressor = joblib.load(regressor_path)
    pipeline_path = Model.get_model_path('DummyPipe')
    pipelines = joblib.load(pipeline_path)
    
def run(raw_data):
    
    # get input data
    data = json.loads(raw_data)
    
    # transform with pipeline
    X = pipelines['feature_pipe'].transform(pd.DataFrame(data))
    
    # make prediction
    y = model.predict(X)
    
    # predict duration
    y_dur = regressor.predict(X)
    
    response = [
        {
            'Products':
            [ 
                pipelines['target_pipe'].transformer_list[1][1].named_steps['target_encode'].col_cats[0][i] 
                for i in range(y.shape[1]) if y[j,i] == 1 
            ],
            'Duration':
                 y_dur[j,0]
        }        
            for j in range(y.shape[0])
    ]

    return response
