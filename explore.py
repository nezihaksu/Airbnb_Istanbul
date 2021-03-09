import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

DF = r'/content/drive/MyDrive/listings.csv'
FILE_TYPE = "csv"


class Explore():
  """Explore the dataset."""
  def __init__(self,df,file_type:str):
    if file_type == "xlsx" or  file_type == "xls":
      self.df = pd.read_excel(df,engine="python")
    self.df = pd.read_csv(df,engine="python")
    self.file_type = file_type

  def __call__(self):
    return self.df

  def intro(self):
    return "===INFO===",self.df.info(),"===DESCRIPTION===",self.df.describe(),"===DTYPES==",self.df.dtypes
  
  def unique_values(self):
    #Unique values that are in features.
    for column in self.df.columns:
      print(column.upper()+ " UNIQUE VALUES")
      print(str(df[column].unique())+"\n")

  def missing_values(self):
   missing_percentage = self.df.isnull().sum()*100/len(self.df)
   plt.figure(figsize=(5, 15))
   missing_percentage.plot(kind='barh')
   plt.xticks(rotation=90, fontsize=10)
   plt.yticks(fontsize=5)
   plt.xlabel("Missing Percentage", fontsize=14)
   plt.show()
   
  #Plotting histograms of the numerical features to see the distribution of each of them.
  def dtype_histogram(self,data_type:str):
    numerical_features = self.df.dtypes[self.df.dtypes == data_type].index.to_list()
    self.df[numerical_features].hist(bins = 50,figsize = (20,15))
    plt.show()

  def corr_heat_map(self):
    pass

