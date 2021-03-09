from explore import Explore
from cleaner import Cleaner
from preprocess import Preprocess
import numpy as np

DF = r'C:\Users\nezih\Desktop\data\listings.csv'
FILE_TYPE = "csv"
IMPUTE = True
INPLACE = True
ALLOWED_NAN_PERCENTAGE = 10
DROP_KEYWORDS = ["code","zipcode","link","url","id","name","thumbnail","picture","pic","description","note"]
NONE_VALUES = [np.nan,None,"None","Null","NONE","NULL","none","null","nan",""," ",0]


class HighLevel():

	@staticmethod
	def cleaner_pipeline(df,file_type,inplace,missing_percentage):
		cleaner = Cleaner(df,file_type)
		cleaner.drop_column_contains(DROP_KEYWORDS)
		cleaner.drop_sentence_columns(inplace=inplace)
		cleaner.drop_date_columns(inplace=inplace)
		cleaner.drop_missing_columns(missing_percentage)
		cleaner.strip_signs()
		df = cleaner.drop_special_columns(inplace=inplace)
		return df 	
	
	@staticmethod
	def preprocess_pipeline(df):
		preprocess = Preprocess(df)
		preprocess.one_hot_encoder()

		
if __name__ == '__main__':
	all_pipelines = HighLevel()
	cleaned_df = all_pipelines.cleaner_pipeline(DF,FILE_TYPE,INPLACE,ALLOWED_NAN_PERCENTAGE)
	preprocessed_df = all_pipelines.preprocess_pipeline(cleaned_df)

