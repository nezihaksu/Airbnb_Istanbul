import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
SEED = 42

class Preprocess():
  """Preprocess the dataset after cleaning."""
  def __init__(self,df):
    self.df = df
    self.categorical_features,self.numerical_features = self._cat_num_features()

  def __call__(self):
    return self.df
  
  def _cat_num_features(self):
    continuous_features = []
    discrete_features = []
    for column in self.df.columns:
      if self.df[column].dtype != "object":
        continuous_features.append(column)
      else:
        discrete_features.append(column)
    return discrete_features,continuous_features

  def features_target(self,target_name:str=None,none_values):
    self.df = self.df[self.df[target_name] is not in none_values]
    target = self.df[target_name]
    self.df = self.df.drop(target_name,axis=1,inplace=True)
    self.categorical_features,self.numerical_features = self._cat_num_features()
    return target

  def drop_multicoll_columns(self,allowed_corr_percentage:int):
    corr_matrix = self.df[self.numerical_features].corr()
    percentage_condition = ((allowed_corr_percentage < corr_matrix.values)&(corr_matrix.values < 1))
    #Finding features that have correlation more than allowed percentage with others.
    corr_features = list(set([corr_matrix.index[row] for row,_ in zip(*np.where(percentage_condition))]))
    self.df.drop(corr_features,axis=1,inplace=True)
    self.categorical_features,self.numerical_features = self._cat_num_features()
    return self.df

  def imputer(self,strategy="most_frequent"):
    simple_imputer = SimpleImputer(strategy=strategy)
    for column in self.df.columns:
      if pd.DataFrame.any(self.df[column].isnull()):
        self.df[column] = simple_imputer.fit_transform(self.df[column].values.reshape(-1,1))
    return self.df

  def one_hot_encoder(self):
    all_categories = []
    for column in self.categorical_features:
      all_categories += [list(self.df[column].unique())]
    encoder = OneHotEncoder(categories = all_categories,sparse=False,handle_unknown='error')
    encoder.fit(self.df[self.categorical_features])
    encoded_categorical_matrix = encoder.transform(self.df[self.categorical_features])
    encoded_categorical_df = pd.DataFrame(encoded_categorical_matrix)
    self.df = pd.concat([encoded_categorical_df,self.df[self.numerical_features]],axis=1)
    return self.df

  def polytrans(self,columns):
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    self.df[columns] = poly_transformer.fit_transform(self.df[columns])
    return self.df

  #If the size of dataset is less than 1000 outlier's effect can be seen in the model's outcome.
  def drop_outliers(self,column:str,upper_quantile:float=0.99,lower_quantile:float=0.01):
    upper_quantile,lower_quantile = self.df[column].quantile(upper_quantile),self.df[column].quantile(lower_quantile)
    self.df = self.df[(df[column] < upper_quantile) & (df[column] > lower_quantile)]
    self.categorical_features,self.numerical_features = self._cat_num_features()
    return self.df

  def train_test_split(self,x,y,test_size,validation=False):
    x_train,y_train,x_test,y_test = train_test_split(x,y,test_size=test_size,random_state=SEED)
    if validation:
      x_test,y_test,x_validation,y_validation = train_test_split(x_test,y_test,test_size=test_size,random_state=SEED)
      return x_train,y_train,x_test,y_test,x_validation,y_validation
    return x_train,y_train,x_test,y_test    

  def stratified_split(self,x,y):
    pass

  def normalization(self,data):
    mean = data.mean()
    std = data.std()
    z = (data - mean) / std
    return z


