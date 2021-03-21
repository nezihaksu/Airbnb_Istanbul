# EDA,Preprocessing and Price Prediction with Regression Models

## Explatory Data Analysis

Firstly,checking the state of the dataset.There are so many null type and text data that won't be useful for regression for now.

First Columns of Raw Dataset:

![](/images/raw_dataset.JPG)

Percentage of Missing Value in Each Feature:

![](/images/missing_percentage_table.JPG)

## Cleaning

As there are so many data types and null values there are also features that are not useful at all.In order automatically ready and prepare dataset by eliminating these features i wrote these functions:

These functions drops columns that are meaningless for our purpose.

First function is a special function that helps to detect given regular expression in all 
cells in a given pandas series object.

drop_missing_columns first makes sure that there are not any columns that have all null values.Secondly it drops the columns that have missing values above certain threshold given by the user.

drop_sentence_columns provides common sentence regex syntax to drop function.

drop_date_columns also provides common date regex syntax to drop function.

drop_special_columns provides pattern of meaningless data.

drop_columns_contains drops the columns according to their names if contains keywords that are in provided list by the user.

strip_signs function drops metric and monatery signs from the numerical data.While doing so converts them into numerical values due to them having object dtype because of signs.Converting them into numerical data proves much help further in the preprocessing pipeline.

```python
	def _drop_type_column(self,pattern:str,inplace:bool):
	  for column in self.df.columns:
	    if any(self.df[column].astype(str).str.contains(pattern,regex=True)):
	      self.df.drop(column,axis=1,inplace=inplace)
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

	def strip_signs(self):
	  num_pattern = r"[0-9]"
	  non_num_pattern = r"[^0-9]"
	  for column in self.df.columns:
	    if all(self.df[column].astype(str).str.contains(num_pattern,regex=True)):
	      self.df[column].replace(non_num_pattern,"",regex=True,inplace=True)
	      self.df[column] = pd.to_numeric(self.df[column])
	  return self.df
```
### NOTE:These function are for preparing a dataset for a machine learning model quickly by automating the process.


## Preprocessing

Before preprocessing thinking in terms of the model's need gives lots of hints to what kind of functions should be created in order to harmonize them with the model creating phase.

Model creating is not linear as one would have to go back to preprocessing phase to drop column or engineer features after seeing performance of the created model.

