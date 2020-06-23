
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import numpy as np

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, dtype):
        self.attribute_names = attribute_names
        self.dtype = dtype
    def fit(self, X, y=None):
        return self        
    def transform(self, X):
        return X[self.attribute_names].astype(self.dtype).values

class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, delimiter=None):
        self.delimiter = delimiter
    def fit(self, X, y=None):
        self.col_cats = {}
        for col in range(X.shape[1]):
            cats = set()
            for row in range(X.shape[0]):
                if self.delimiter:
                    for cat in X[row,col].split(self.delimiter):
                        if not cat.strip() == '':
                            cats.add(cat.strip())
                else:
                    cats.add(X[row,col])
            self.col_cats[col] = list(cats)
        return self
    def transform(self, X):
        X_tr = []
        for col in range(X.shape[1]):
            X_enc = np.zeros([X.shape[0], len(self.col_cats[col])])
            for row in range(X.shape[0]):
                if self.delimiter:
                    cats = str(X[row,col]).split(self.delimiter)
                    for col_cat_idx in range(len(self.col_cats[col])):
                        if self.col_cats[col][col_cat_idx] in cats:
                            X_enc[row, col_cat_idx] = 1
                else:
                    for col_cat_idx in range(len(self.col_cats[col])):
                        if self.col_cats[col][col_cat_idx] == X[row,col]:
                            X_enc[row, col_cat_idx] = 1
            X_enc = np.array(X_enc)
            X_tr.append(X_enc)
        X_tr = np.concatenate(X_tr, axis=1)
        return X_tr
    
def create_pipelines(cfg):
    
    # Pipeline for multilabel features
    multi_pipe = Pipeline([
        ('multi_feat_select', DataFrameSelector(cfg['multi_cols'], str)),
        ('multi_encode', MultiHotEncoder(delimiter=' '))
    ])
    
    # combine features
    feat_union = FeatureUnion([
        ('multi_features', multi_pipe)
    ])
    
    # preprocess all features
    all_feat_pipe = Pipeline([
        ('all_features_pipe', feat_union),
#         ('all_feautres_pca', PCA(n_components=0.8, svd_solver = 'full'))
    ])
    
    # Pipeline for multi target cols
    multi_target_pipe = Pipeline([
        ('target_select', DataFrameSelector(cfg['multi_target_cols'], str)),
        ('target_encode', MultiHotEncoder(delimiter=' '))
    ])

    # Pipeline for numerical target cols
    num_target_pipe = Pipeline([
        ('num_feature_select', DataFrameSelector(cfg['num_target_cols'], float))
    ])
    
    all_target_pipe = FeatureUnion([
        ('num_targets', num_target_pipe),
        ('multi_targets', multi_target_pipe)
    ])

    return { 'feature_pipe': all_feat_pipe, 'target_pipe': all_target_pipe }
