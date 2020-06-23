
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
    
def create_pipeline(cfg):    
    # Pipeline for multilabel features
    multi_pipe = Pipeline([
        ('multi_feat_select', DataFrameSelector(cfg['multi_cols'], str)),
#         ('multi_replace_missing', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=tuple())),
        ('multi_encode', MultiHotEncoder(delimiter=' '))
    ])
    
    # Pipeline for target features
    target_pipe = Pipeline([
        ('target_select', DataFrameSelector(cfg['target_cols'], str)),
#         ('multi_replace_missing', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=tuple())),
        ('target_encode', MultiHotEncoder(delimiter=' '))
    ])

#   # Pipeline for categories
#     cat_pipe = Pipeline([
#         ('cat_feature_select', DataFrameSelector(cfg['cat_cols'])),
#         ('cat_replace_missing', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='0')),
#         ('cat_one_hot_encode', OneHotEncoder(sparse=False))
#     ])

    # Pipeline for numericals
    num_pipe = Pipeline([
        ('num_feature_select', DataFrameSelector(cfg['num_cols'], float))
#         ('num_replace_missing', SimpleImputer(missing_values=np.nan, strategy='mean')),
#         #('num_normalization', MinMaxScaler())
#         ('num_standardization', StandardScaler())
    ])

    feat_union = FeatureUnion([
#         ('num_features', num_pipe),
#         ('cat_features', cat_pipe),
        ('multi_features', multi_pipe)
    ])
    
    all_feat_pipe = Pipeline([
        ('all_features_pipe', feat_union),
#         ('all_feautres_pca', PCA(n_components=0.8, svd_solver = 'full'))
    ])
    
    pipeline = FeatureUnion([
        ("all_feat_pipe", all_feat_pipe),
        ('num_targets', num_pipe),
        ("target_pipe", target_pipe)
    ])

    return pipeline
