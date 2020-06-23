
import numpy as np
import pandas as pd
import json
import os
import joblib
from lookup import LookUpClassifier
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('LookUpModel')
    model = joblib.load(model_path)
    
def run(raw_data):
    
    # get input data
    data = json.loads(raw_data)
    
    X = [ [d['ProductId'], d['Country'], ' '.join(d['Symptoms']) ] for d in data]
    
    # make prediction
    y = model.predict(X)
    
    response = [
        {
            'Products':
                y[j].split(' '),
            'Duration':
                 round(np.random.random())*10 #y_dur[j,0]
        }        
            for j in range(len(y))
    ]

    return response
