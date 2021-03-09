import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

class Preprocess():

  def __init__(self,df):
    self.df = df
    self.numerical_df,self.categorical_df = self._split_into_cat_num_df()
  
  def __call__(self):
    return self.df
  
  def _split_into_cat_num_df(self):
    num_pattern = r"[0-9]"
    continuous_features = []
    discrete_features = []
    for column in self.df.columns:
      if all(self.df[column].astype(str).str.contains(num_pattern,regex=True)):
        continuous_features.append(column)
      else:
        discrete_features.append(column)
    return self.df[continuous_features],self.df[discrete_features]


  def drop_multicoll_columns(self,allowed_corr_percentage:int):
    corr_matrix = self.numerical_df.corr()
    corr_matrix[corr_matrix]
    #multicoll_indexes = np.where(np.logical_and(corr_matrix < 1.0, corr_matrix > self.corr_percetage))
    return corr_matrix


  def one_hot_encoder(self):
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
      return self.df
  def polytrans(self):
    pass