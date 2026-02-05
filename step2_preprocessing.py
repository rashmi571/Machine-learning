# step : 2 preprossing (data cleaning)
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("/content/sample_data/student_performance_data.csv")

print("Missing values")
print(df.isnull().sum())

le=LabelEncoder()
df['internet_use']=le.fit_transform(df['internet_use'])
df['passed']=le.fit_transform(df['passed'])

print("After label encoding")
print(df.head())

print("Data type")
print(df.dtypes)
