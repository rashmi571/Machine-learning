import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error

df=pd.read_csv("/content/sample_data/creditcard.csv")

x=df.drop(['Class'],axis=1)
y=df['Class']

x_train, y_train, x_test, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestRegressor(n_estimators=200,random_state=42)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print('Mean abasolute Error: ',np.round(mae,2))
print('Mean squared Error: ',np.round(mse,2))
print('Root Mean squared Error: ',np.round(rmse,2))
print('r2 Error: ',np.round(r2,2))


