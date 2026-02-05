#step 1 load and understanding data
import pandas as pd
#load
df=pd.read_csv("/content/sample_data/student_performance_data.csv")

print("sample rows")
print(df.head())

print("\nDataSet shape")
print(f"Rows: {df.shape[0]},Columns: {df.shape[1]}")

print("\n Data information")
print(df.info())

print("\nData summery statistics")
print(df.describe(include='all'))

print("\nData types")
print(df.dtypes)

print("Messing values")
print(df.isnull().sum())

