import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

class Preprocess():
  """Preprocess the dataset after cleaning."""
  def __init__(self,df):
    self.df = df
    self.numerical_df,self.categorical_df,self.numerical_features,self.categorical_features = self._split_into_cat_num_df()
      
  def __call__(self):
    return self.df
  
  def _split_into_cat_num_df(self):
    num_pattern = r"[0-9]"
    continuous_features = []
    discrete_features = []
    for column in self.df.columns:
      if all(self.df[column].astype(str).str.contains(num_pattern,regex=True)) and len(self.df[column].unique()) > 2:
        continuous_features.append(column)
      else:
        discrete_features.append(column)
    return self.df[continuous_features],self.df[discrete_features],continuous_features,discrete_features

  def drop_multicoll_columns(self,allowed_corr_percentage:int):
    corr_matrix = self.numerical_df.corr()
    corr_matrix[corr_matrix]
    #multicoll_indexes = np.where(np.logical_and(corr_matrix < 1.0, corr_matrix > self.corr_percetage))
    return corr_matrix


  def imputer(self,strategy="most_frequent"):
    simple_imputer = SimpleImputer(strategy=strategy)
    for column in self.df.columns:
      if pd.DataFrame.any(self.df[column].isnull()):
        self.df[column] = simple_imputer.fit_transform(self.df[column].values.reshape(-1,1))
    print(self.df.describe())
    return self.df


  def one_hot_encoder(self):
    one_hot = OneHotEncoder(handle_unknown="ignore")
    #Preprocessing for numerical and categorical data
    transformer = ColumnTransformer(
      transformers=[('one_hot',one_hot,self.categorical_features)
      ])
    transformer.fit_transform(self.categorical_df)
    return self.df

  def polytrans(self):
    pass
