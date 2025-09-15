#Create classes with method to remove columns and transform columns
from sklearn.base import BaseEstimator,TransformerMixin

from pathlib import Path
import os
import sys
from prediction_model.config import config

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
import numpy as np

#Drop columns not needed in X
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self,variables_to_drop=None):
        self.variables_to_drop = variables_to_drop
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X=X.drop(columns = self.variables_to_drop)
        return X

#Create new column as sum of 4 other relevant numerical predictors    
class DomainProcessing(BaseEstimator, TransformerMixin):
    def __init__(self,variable_to_add = None):
        self.new_column = config.NEW_FEATURE_ADD
        self.variable_to_add = variable_to_add
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X[self.new_column] = X[self.variable_to_add].sum(axis=1)
        return X
    
#Encode the text based predictors as 1 or 0 depending on values    
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables = None):
        self.variables = variables

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        for column_name, positive_values in self.variables.items():
            X[column_name] = X[column_name].apply(lambda x: 1 if x.strip() in positive_values else 0)
        return X

# Log transform the large numerical columns
class LogTransforms(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col])
        return X

        
