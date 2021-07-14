from explore import Explore
from cleaner import Cleaner
from preprocess import Preprocess
from model import MultiRegression
import numpy as np

#Open the file.
DF = r'C:\Users\nezih\Desktop\data\listings.csv'
FILE_TYPE = "csv"
#Cleaning.
IMPUTE = True
INPLACE = True
ALLOWED_NAN_PER = 10
DROP_KEYWORDS = ["code","zipcode","link","url","id","name","thumbnail","picture","pic","description","note"]
NONE_VALUES = [np.nan,None,"None","Null","NONE","NULL","none","null","nan",""," "]
#Preprocess
TARGET = "price"
OUTLIER_COLUMN = None
POLYTRANS_COLUMNS = None
UPPER_QUANTILE = 0.99
LOWER_QUANTILE = 0.01
ALLOWED_CORR_PER = 0.8
TEST_SIZE = 0.25
VALIDATION_BOOL = False
STRATIFIED_SPLIT = False
#Model training.
LEARNING_RATE = 0.01
N_ITERS = 100
BATCH_SIZE = 100
DECAY_RATE = 0.9
TOLERANCE = 1e-06


class Pipelines():

	def __init__(self,df,file_type=None):
		self.df = df
		self.file_type = file_type

	def explorer(self,x=None,y=None):
		explorer = Explore(self.df,self.file_type)
		explorer.intro()
		explorer.unique_values()
		explorer.missing_values()
		#explorer.dtype_histogram()
		#explorer.scatter_plot(x,y)

	def cleaner_pipeline(self,drop_keywords,inplace=None,missing_percentage=10):
		cleaner = Cleaner(self.df,self.file_type)
		cleaner.drop_column_contains(drop_keywords)
		cleaner.drop_sentence_columns(inplace=inplace)
		cleaner.drop_date_columns(inplace=inplace)
		cleaner.drop_missing_columns(missing_percentage)
		cleaner.strip_signs()
		cleaner.drop_special_columns(inplace=inplace)
		df = cleaner.space_to_underscore()
		return df
	
	def preprocess_pipeline(self,df,upper_quantile,lower_quantile,target_name,none_values,outlier_column:str=None,
							polytrans_columns:list=None,corr_percentage=0.7,test_size=0.25,validation=False):
		preprocess = Preprocess(df)
		target_feature = preprocess.target_feature(none_values,target_name)
		if outlier_column != None:
			preprocess.drop_outliers(outlier_column,upper_quantile,lower_quantile)
		preprocess.drop_multicoll_columns(ALLOWED_CORR_PER)
		preprocess.imputer()
		if polytrans_columns != None:
			preprocess.polytrans(polytrans_columns)
		features = preprocess.one_hot_encoder()
		if validation:
			x_train,y_train,x_test,y_test,x_validation,y_validation = train_test_split(features,target_feature,test_size,validation)
			return x_train,y_train,x_test,y_test,x_validation,y_validation
		x_train,y_train,x_test,y_test = preprocess.train_test_split(features,target_feature,test_size,validation)
		return x_train,x_test,y_train,y_test

	def model_pipeline(self,x_train,y_train,x_test,y_test,learning_rate,n_iters,batch_size,decay_rate,tolerance):
		mr = MultiRegression(learning_rate,n_iters,batch_size,decay_rate,tolerance)
		mr.fit(x_train,y_train)
		y_pred = mr.predict(x_test)
		return mr.r2_score(y_test,y_pred),model.mse_score(y_test,y_pred)
		
if __name__ == '__main__':
	pipelines = Pipelines(DF,FILE_TYPE)
	expore = pipelines.explorer()
	cleaned_df = pipelines.cleaner_pipeline(DROP_KEYWORDS,INPLACE,ALLOWED_NAN_PER)
	x_train,x_test,y_train,y_test = pipelines.preprocess_pipeline(
																df=cleaned_df,
																outlier_column = OUTLIER_COLUMN,
																polytrans_columns = POLYTRANS_COLUMNS,
																corr_percentage = ALLOWED_CORR_PER,
																target_name = TARGET,
																upper_quantile = UPPER_QUANTILE,
																lower_quantile = LOWER_QUANTILE,
																none_values = NONE_VALUES,
																test_size = TEST_SIZE,
																validation = VALIDATION_BOOL)
	r2_score,mse_score = pipelines.model_pipeline(x_train,y_train,x_test,y_test,LEARNING_RATE,N_ITERS,BATCH_SIZE,DECAY_RATE,TOLERANCE)
	print(r2_score,mse_score)
