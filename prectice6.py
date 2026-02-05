#price prediction of house
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

#load the data
df=pd.read_csv("/content/sample_data/house_price_data_100.csv")

#understand the data
print("Rows and columns in Data set")
print(df.head())

print("\n Data Information")
print(df.info())

print("\n No. of rows and columns")
print(f'Rows: {df.shape[0]},Columns: {df.shape[1]}')

print("\n Summery of data")
print(df.describe(include='all'))

print("\n Missing values")
print(df.isnull().sum())

#define the features of the data set
features=['area_sqft','bedrooms','bathrooms','house_age','location_score']

#give data for train and test
x=df[features ]
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train model using linearregression
model=LinearRegression()
model.fit(x_train,y_train)

#test prediction
y_predict=model.predict(x_test)

#Evaluation
mae=mean_absolute_error(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_predict)

print("Mean absulate error: ",np.round(mae,2))
print("Mean square error: ",np.round(mse,2))
print("Root mean square error: ",np.round(rmse,2))
print("R2 score: ",np.round(r2,2))

#user prediction
print("-------price prediction--------")
try:
  Area_sqft=int(input("Enter house area sqft: "))
  Bedrooms=int(input("Enter number of bedrooms: "))
  Bathrooms=int(input("Enter number of bathrooms: "))
  House_age=int(input("Enter house age: "))
  Location_score=int(input("Enter location score: "))

  user_input = pd.DataFrame(
        [[Area_sqft, Bedrooms, Bathrooms, House_age, Location_score]],
        columns=['area_sqft', 'bedrooms', 'bathrooms', 'house_age', 'location_score']
    )

  model_predict=model.predict(user_input)

  print(f"Predicted house price: {np.round(model_predict[0],2)}")

except Exception as e:
  print("An any error",e)