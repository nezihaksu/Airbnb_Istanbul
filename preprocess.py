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
    num_pattern = r"[\d]"
    continuous_features = []
    discrete_features = []
    for column in self.df.columns:
      if self.df[column].dtype != "object":
        continuous_features.append(column)
      else:
        discrete_features.append(column)
    return self.df[continuous_features],self.df[discrete_features],continuous_features,discrete_features

  def drop_multicoll_columns(self,allowed_corr_percentage:int):
    corr_matrix = self.df[self.numerical_features].corr()
    percentage_condition = ((allowed_corr_percentage < corr_matrix.values)&(corr_matrix.values < 1))
    #Finding features that have correlation more than allowed percentage with others.
    corr_features = list(set([corr_matrix.index[row] for row,_ in zip(*np.where(percentage_condition))]))
    self.df.drop(corr_features,axis=1,inplace=True)
    return self.df

  def imputer(self,strategy="most_frequent"):
    simple_imputer = SimpleImputer(strategy=strategy)
    for column in self.df.columns:
      if pd.DataFrame.any(self.df[column].isnull()):
        self.df[column] = simple_imputer.fit_transform(self.df[column].values.reshape(-1,1))
    return self.df

  def one_hot_encoder(self):
    encoder = OneHotEncoder(categories = self.categorical_features,handle_unknown='error')
    #encoded_categorical_df = encoder.fit_transform(self.df[self.categorical_features])
    #self.df = pd.concat([encoded_categorical_df,self.df[self.numerical_features]])
    return self.df

  def polytrans(self):
    pass

  #If the size of dataset is less than 1000 outlier's effect can be seen in the model's outcome.
  def drop_outliers(self,column:str,upper_quantile:float=0.99,lower_quantile:float=0.01):
    upper_quantile,lower_quantile = self.df[column].quantile(upper_quantile),self.df[column].quantile(lower_quantile)
    self.df = self.df[(df[column] < upper_quantile) & (df[column] > lower_quantile)]
    return self.df

