# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# pd.set_option('display.max_columns', None)
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.impute import SimpleImputer
# import re

class HighLevel():

	def __init__(self,df,file_type):
		self.df = df
		self.file_type = file_type


	def cleaner_pipeline(self):
		cleaner = Cleaner(self.df,self.file_type)
		cleaner.drop_column_contains(DROP_KEYWORDS)
		cleaner.drop_sentence_columns(inplace=True)
		cleaner.drop_date_columns(inplace=True)
		cleaner.drop_missing_columns(10)
		cleaner.strip_signs()
		cleaner.drop_special_columns(True)
		df = cleaner.imputer()
		return df
	
	
	def preproces_pipeline():
		pass


