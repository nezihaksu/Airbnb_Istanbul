from .explore import Explore
from .cleaner import Cleaner
from .preprocess import Preprocess

DF = r'C:\Users\nezih\Desktop\data\listings.csv'
FILE_TYPE = "csv"
IMPUTE = True
ALLOWED_NAN_PERCENTAGE = 10
DROP_KEYWORDS = ["code","zipcode","link","url","id","name","thumbnail","picture","pic","description","note"]
NONE_VALUES = [np.nan,None,"None","Null","NONE","NULL","none","null","nan",""," ",0]


class HighLevel():

	def cleaner_pipeline(df,file_type):
		cleaner = Cleaner(df,file_type)
		cleaner.drop_column_contains(DROP_KEYWORDS)
		cleaner.drop_sentence_columns(inplace=True)
		cleaner.drop_date_columns(inplace=True)
		cleaner.drop_missing_columns(10)
		cleaner.strip_signs()
		cleaner.drop_special_columns(True)
		df = cleaner.imputer()
		return df 	
	
	
	def preprocess_pipeline(df):
		preprocess = Preprocess(df)
		preprocess.one_hot_encoder()

		
if __name__ == '__main__':
	all_pipelines = HighLevel()
	cleaned_df = all_pipelines.cleaner_pipeline(DF,FILE_TYPE)
	preprocessed_df = all_pipelines.preprocess_pipeline(cleaned_df)

