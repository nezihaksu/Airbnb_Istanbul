# EDA,Preprocessing and Price Prediction with Regression Models

## Explatory Data Analysis

Firstly,checking the state of the dataset.There are so many null type and text data that won't be useful for regression for now.

Columns of Raw Dataset:

![](/images/raw_dataset.JPG)

Percentage of Missing Value in Each Feature:

![](/images/missing_percentage_table.JPG)


## Cleaning

As there are so many data types and null values there are also features that are not useful at all.In order to automatically ready and prepare dataset by eliminating these features i wrote these functions:

These functions drop columns that are meaningless for our purpose.

First function is a special function that helps to detect given regular expression in all cells in a given pandas series object.

drop_missing_columns first makes sure that there are not any columns that have all null values.Secondly it drops the columns that have missing values above certain threshold given by the user.

drop_sentence_columns provides common sentence regex syntax to drop function.

drop_date_columns also provides common date regex syntax to drop function.

drop_special_columns provides pattern of meaningless data.

drop_columns_contains drops the columns according to their names if contains keywords that are in provided list by the user.

strip_signs function drops metric and monatery signs from the numerical data.While doing so converts them into numerical values due to them having object dtype because of signs.Converting them into numerical dtype proves much help further in the preprocessing pipeline.


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

Dataset after cleaned:

![](/images/cleaned_dataset.JPG)

## Preprocessing

Before preprocessing,thinking in terms of the model's needs gives lots of hints to what kind of functions should be created in order to harmonize them with the model creating phase.

Modelling is not linear as one would have to go back to preprocessing phase to drop column or engineer features after seeing performance of the created model.

For automation purposes separating dataset into numerical and categorical features helps to transform them separately.

```python
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
```
### Multicollinearity

For linear regression dropping correlated features is vital because multicollinearity causes coefficients to be insignificant because of increased variance.

So,to be able to interpret our linear regression model i wrote a function that drops features that are correlated above certain percentage threshold that the user has given:

```python
  def drop_multicoll_columns(self,allowed_corr_percentage:int):
    corr_matrix = self.df[self.numerical_features].corr()
    percentage_condition = ((allowed_corr_percentage < corr_matrix.values)&(corr_matrix.values < 1))
    #Finding features that have correlation more than allowed percentage with others.
    corr_features = list(set([corr_matrix.index[row] for row,_ in zip(*np.where(percentage_condition))]))
    self.df.drop(corr_features,axis=1,inplace=True)
    self.categorical_features,self.numerical_features = self._cat_num_features()
    return self.df
```

Also updating dataframe,categorical features and numerical features lists after dropping any kind of column must be not be forgotten.

### Outliers

Outliers are tricky in the sense of they may hold important information regarding dataset but also may distrupt model's predictions when there are not many instances to fit the model on.

In this case becase the dataset have number of instances over 10.000 it is not necessary to drop outliers at all since model can involve them to extract useful information out of them with this much instances.

In any case including an outlier dropping function would prove useful for another datasets.

```python
  def drop_outliers(self,column:str,upper_quantile:float=0.99,lower_quantile:float=0.01):
    upper_quantile,lower_quantile = self.df[column].quantile(upper_quantile),self.df[column].quantile(lower_quantile)
    self.df = self.df[(df[column] < upper_quantile) & (df[column] > lower_quantile)]
    self.categorical_features,self.numerical_features = self._cat_num_features()
    return self.df

```
### Imputer

Using most_frequent imputing strategy helps to impute both categorical and numerical data.

```python
  def imputer(self,strategy="most_frequent"):
    simple_imputer = SimpleImputer(strategy=strategy)
    for column in self.df.columns:
      if pd.DataFrame.any(self.df[column].isnull()):
        self.df[column] = simple_imputer.fit_transform(self.df[column].values.reshape(-1,1))
    return self.df

```

### Polynomial Transformation

In order to make the relationship between the independent and dependent variable more apparent taking squares or even cubes of the independent variables may prove useful.

This can be decided according particular independent variable and dependent variable scatter plot.

```python
  def polytrans(self,columns):
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    self.df[columns] = poly_transformer.fit_transform(self.df[columns])
    return self.df
```

### One Hot Encoding

This function creates dummy variables out of categorical features.

```python
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
```
