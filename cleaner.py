import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
pd.set_option('display.max_columns', None)

DF = r'C:\Users\nezih\Desktop\data\listings.csv'
FILE_TYPE = "csv"
IMPUTE = True
ALLOWED_NAN_PERCENTAGE = 10
DROP_KEYWORDS = ["code","zipcode","link","url","id","name","thumbnail","picture","pic","description","note"]
NONE_VALUES = [np.nan,None,"None","Null","NONE","NULL","none","null","nan",""," ",0]

class Cleaner():
	"""Clean the dataset to visualize better."""
	def __init__(self,df,file_type:str):
	  if file_type == "xlsx" or  file_type == "xls":
	    self.df = pd.read_excel(df,engine="python")
	  self.df = pd.read_csv(df,engine="python")
	  self.file_type = file_type

	def __call__(self):
	  return self.df

	def _drop_type_column(self,pattern:str,inplace:bool):
	  for column in self.df.columns:
	    if any(self.df[column].astype(str).str.contains(pattern,regex=True)):
	      self.df.drop(column,axis=1,inplace=inplace)
	  return self.df

	#Expanding one column dataframe into multiple columns according to split character.
	def split_column_into_df(self,column_index:int,split_char:str):
	  if len(df.columns) == 1:
	    quotes_strip = list(self.df.columns)[0].replace(strip_char,'')
	    columns_split = quotes_strip.split(split_char)
	    self.df = self.df[self.df.iloc[:,0].name].str.split(pat = split_char,expand = True)
	    self.df.columns =  columns_split
	    self.df.replace(split_char,'',regex = True,inplace = True)
	  print("This method is only for explanding single column dataframes!")
	  return self.df

	def drop_missing_columns(self,percentage):
	  self.df.dropna(how="all",axis=1,inplace=True)
	  #In case of dropna method does not work as expect because of value type \
	  #this loop over columns would solve some of the problems.
	  for column in self.df.columns:
	    if len(self.df[column].unique()) == 1:
	      self.df.drop(column,axis=1,inplace=True)
	  missing_percentage = self.df.isnull().sum()*100/len(self.df)
	  features_left = missing_percentage[missing_percentage < percentage].index
	  self.df = self.df[features_left] 
	  return self.df

	#Drop columns by their names.
	def drop_column_contains(self,keywords:list):
	  for keyword in keywords:
	    keyword_pattern = re.compile(keyword)
	    for column in self.df.columns:
	      if keyword_pattern.search(column):
	        self.df.drop(column,axis=1,inplace=True)
	  return self.df

	def drop_sentence_columns(self,inplace):
	  #sentence_pattern = r'[A-z][A-z]+?\W'
	   sentence_pattern = r'(\w \w){2}'
	   link_pattern = r'[A-z][A-z]+?://'
	   text_pattern = r'|'.join((sentence_pattern,link_pattern))  
	   return self._drop_type_column(text_pattern,inplace)
 
	def drop_date_columns(self,inplace:bool):
	  date_pattern_dash = r"([12]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]))"
	  date_pattern_dot = r"([12]\d{3}.(0[1-9]|1[0-2]).(0[1-9]|[12]\d|3[01]))"
	  date_pattern_slash = r"([12]\d{3}/(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01]))"
	  date_pattern_dash_text =  r"([12]\d{3}-([A-z]+)-(0[1-9]|[12]\d|3[01]))"
	  date_pattern_dot_text = r"([12]\d{3}.([A-z]+).(0[1-9]|[12]\d|3[01]))"
	  date_pattern_slash_text = r"([12]\d{3}/([A-z]+)/(0[1-9]|[12]\d|3[01]))"
	  date_pattern = r'|'.join((date_pattern_dash,
	                            date_pattern_dot,
	                            date_pattern_slash,
	                            date_pattern_dash_text,
	                            date_pattern_dot_text,
	                            date_pattern_slash_text))
	  return self._drop_type_column(date_pattern,inplace)

	def drop_special_columns(self,inplace:bool):
	  starts_with_special_pattern = r'^[^\w]'
	  ends_with_special_pattern = r'[^\w]^'
	  starts_ends_special_pattern =  r'|'.join((starts_with_special_pattern,ends_with_special_pattern))
	  return self._drop_type_column(starts_ends_special_pattern,inplace)
	
	#When there is a sign near a number that column dtype is object \
	#It should be converted into numerical dtype after stripping for further processes.(int64,float64).
	def strip_signs(self):
	  num_pattern = r"[0-9]"
	  non_num_pattern = r"[^0-9]"
	  for column in self.df.columns:
	    if all(self.df[column].astype(str).str.contains(num_pattern,regex=True)):
	      self.df[column].replace(non_num_pattern,"",regex=True,inplace=True)
	      self.df[column] = pd.to_numeric(self.df[column])
	  return self.df
	
	def space_to_underscore(self):
		self.df.replace(" ","_",regex=True,inplace=True)
		return self.df
