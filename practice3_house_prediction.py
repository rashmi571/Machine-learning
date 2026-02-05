#practice house predict price
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data=pd.read_csv("/content/sample_data/house_price_prediction.csv")



x=data[["size_sqft","bedrooms","bathrooms"]] # Update these column names if necessary
y=data["price"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

predict_price=model.predict(x_test)

mae=mean_absolute_error(y_test,predict_price)
mse=mean_squared_error(y_test,predict_price)
rmse=np.sqrt(mse)
r2=r2_score(y_test,predict_price)

print("Mean absolute error: ",round(mae,2))
print("Mean squared error: ",round(mse,2))
print("Root mean squared error: ",round(rmse,2))
print("R2 score ",round(r2,2))

new_house=pd.DataFrame([[2200,7,6]], columns=x.columns)
new_price_predict=model.predict(new_house)
print(f"Predicted price for \n{new_house} new house: {new_price_predict}")
