#project -1
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
import pandas as pd

data=pd.read_csv("/content/sample_data/studyhours_score.csv")

x=data[["studyhours"]]
y=data["score"]

model=LinearRegression()

#give data for training
model.fit(x,y)

#model prediction
predict_score=model.predict(x)

#evaluate
mas=mean_absolute_error(y,predict_score)
mse=mean_squared_error(y,predict_score)
rmse=np.sqrt(mse)

print("mean absolute error: ",mas)
print("mean squared error: ",mse)
print("root mean squared error: ",rmse)

#optional
new_predict=model.predict([[7]])
print("New predicted score for 7 hours: ",new_predict)

