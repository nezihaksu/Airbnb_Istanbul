from explore import Explore
from cleaner import Cleaner
from preprocess import Preprocess
import numpy as np

DF = r'C:\Users\nezih\Desktop\data\listings.csv'
FILE_TYPE = "csv"
IMPUTE = True
INPLACE = True
ALLOWED_NAN_PER = 10
DROP_KEYWORDS = ["code","zipcode","link","url","id","name","thumbnail","picture","pic","description","note"]
NONE_VALUES = [np.nan,None,"None","Null","NONE","NULL","none","null","nan",""," ",0]
OUTLIER_COLUMN = ""
ALLOWED_CORR_PER = 0.8

class Pipelines():

	def cleaner_pipeline(self,df,file_type,inplace,missing_percentage):
		cleaner = Cleaner(df,file_type)
		cleaner.drop_column_contains(DROP_KEYWORDS)
		cleaner.drop_sentence_columns(inplace=inplace)
		cleaner.drop_date_columns(inplace=inplace)
		cleaner.drop_missing_columns(missing_percentage)
		cleaner.strip_signs()
		df = cleaner.drop_special_columns(inplace=inplace)
		return df 	
	
	def preprocess_pipeline(self,df,outlier_column:str,corr_percentage):
		preprocess = Preprocess(df)
		return preprocess.drop_multicoll_columns(ALLOWED_CORR_PER)

if __name__ == '__main__':
	pipelines = Pipelines()
	cleaned_df = pipelines.cleaner_pipeline(DF,FILE_TYPE,INPLACE,ALLOWED_NAN_PER)
	matrix = pipelines.preprocess_pipeline(cleaned_df,OUTLIER_COLUMN,ALLOWED_CORR_PER)
	print(matrix)
	#print(preprocessed_df)
