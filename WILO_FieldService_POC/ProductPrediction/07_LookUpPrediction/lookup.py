
import numpy as np
import pandas as pd

class LookUpClassifier():
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def fit(self, df):
        
        self.product_transform = list(set(x for l in df['ProductNrs'].apply(lambda x: x.split()).values.tolist() for x in l))
        
        symptoms_per_case_df = pd.DataFrame(df['Symptoms'].str.split(' ').tolist(), index=df['Job Card.JobCard Number']).stack().reset_index([0, 'Job Card.JobCard Number'])
        symptoms_per_case_df.columns = ['Job Card.JobCard Number', 'Symptom']

        prodnr_per_case_df = pd.DataFrame(df['ProductNrs'].str.split(' ').tolist(), index=df['Job Card.JobCard Number']).stack().reset_index([0, 'Job Card.JobCard Number'])
        prodnr_per_case_df.columns = ['Job Card.JobCard Number', 'ProductNr']

        df = pd.merge(symptoms_per_case_df, df, on='Job Card.JobCard Number', how='left')
        df = pd.merge(prodnr_per_case_df, df, on='Job Card.JobCard Number', how='left')

        df = df[['ProductId', 'Country', 'Symptom', 'ProductNr']].replace('', np.nan).dropna().reset_index(drop=True)

#         self.model = df.groupby(['ProductId', 'Country', 'Symptom'])#
        self.model = {}
        for i in range(len(df)):
            if not df['ProductId'][i] in self.model:
                self.model[df['ProductId'][i]] = {}
            if not df['Country'][i] in self.model[df['ProductId'][i]]:
                self.model[df['ProductId'][i]][df['Country'][i]] = {}
            if not df['Symptom'][i] in self.model[df['ProductId'][i]][df['Country'][i]]:
                self.model[df['ProductId'][i]][df['Country'][i]][df['Symptom'][i]] = []
            self.model[df['ProductId'][i]][df['Country'][i]][df['Symptom'][i]].append(df['ProductNr'][i])
    
    def predict(self, X):
        # X = [['<prodid>', '<country>', '<symptom1>, <symptom2>']]
        y = []
        for row in X:
            y_row = []
            for symptom in row[2].split(' '):
                if row[0] in self.model:
                    if row[1] in self.model[row[0]]:
                        if symptom in self.model[row[0]][row[1]]:
                            y_row += self.model[row[0]][row[1]][symptom]
            
            y_probs = np.random.random(len(y_row))
            y_row = [ y_row[i] for i in range(len(y_probs)) if y_probs[i] > self.threshold ]
            
            y.append(' '.join(map(str, list(set(y_row)))))
            
        return y
    
    def transform_products(self, y):
        # y = [ '<prod1> <prod2>', '<prod1> <prod3>' ]
        y_tr = np.zeros([len(y), len(self.product_transform)])
        for row in range(len(y)):
            for prod in y[row].split(' '):
                if prod in self.product_transform:
                    y_tr[row, self.product_transform.index(prod)] = 1
        return y_tr
