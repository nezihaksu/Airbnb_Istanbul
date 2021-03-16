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
OUTLIER_COLUMN = None
UPPER_QUANTILE = 0.99
LOWER_QUANTILE = 0.01
ALLOWED_CORR_PER = 0.8



class Pipelines():

	def explorer(self,df,file_type=None,x=None,y=None):
		explorer = Explore(df,file_type)
		explorer.intro()
		explorer.unique_values()
		explorer.missing_values()
		explorer.dtype_histogram()
		explorer.scatter_plot(x,y)

	def cleaner_pipeline(self,df,file_type=None,inplace=None,missing_percentage=10):
		cleaner = Cleaner(df,file_type)
		cleaner.drop_column_contains(DROP_KEYWORDS)
		cleaner.drop_sentence_columns(inplace=inplace)
		cleaner.drop_date_columns(inplace=inplace)
		cleaner.drop_missing_columns(missing_percentage)
		cleaner.strip_signs()
		cleaner.drop_special_columns(inplace=inplace)
		df = cleaner.space_to_underscore()
		return df
	
	def preprocess_pipeline(self,df,upper_quantile,lower_quantile,outlier_column:str=None,corr_percentage=0.7):
		preprocess = Preprocess(df)
		#if outlier_column != None:
		#	preprocess.drop_outliers(outlier_column,upper_quantile,lower_quantile)
		#preprocess.drop_multicoll_columns(ALLOWED_CORR_PER)
		#df = preprocess.imputer()
		#df = preprocess.one_hot_encoder()
		return preprocess.numerical_df



if __name__ == '__main__':
	pipelines = Pipelines()
	cleaned_df = pipelines.cleaner_pipeline(DF,FILE_TYPE,INPLACE,ALLOWED_NAN_PER)
	preprocessed_df = pipelines.preprocess_pipeline(cleaned_df,OUTLIER_COLUMN,ALLOWED_CORR_PER)
	print(preprocessed_df)
	