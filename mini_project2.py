#project 2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("/content/sample_data/Exam_Score_Prediction.csv")

x=data[["study_hours"]]
y=data["exam_score"]

model=LinearRegression()
#model train
model.fit(x,y)

#model predict
predict_score=model.predict(x)

#evaluate
mas=mean_absolute_error(y,predict_score)
mse=mean_squared_error(y,predict_score)
rmse=np.sqrt(mse)
r2=r2_score(y,predict_score)

print("Mean absolute error: ",round(mas,2))
print("Mean squared error: ",round(mse,2))
print("Root mean squared error: ",round(rmse,2))
print("R^2 score: ",round(r2,2))

new_hour=9
predict_new_score=model.predict([[new_hour]])
print(f"Predict score for new {new_hour} hour: {predict_new_score}")

#histogram
plt.figure(figsize=(8,6))
plt.hist(data["exam_score"],bins=10,color='skyblue',edgecolor="black")
plt.xlabel("Exam Score")
plt.ylabel("number of student")
plt.title("Histogram of Exam Scores")
plt.grid(True)
plt.show()

#scatter+Regression line
plt.figure(figsize=(8,6))
plt.scatter(x,y,color="blue",label="Actual score")
plt.plot(x,predict_score,color="red",label="predict score(Regression line)")
plt.xlabel("study hours")
plt.ylabel("exam score")
plt.title("study hours vs exam score")
plt.grid(True)
plt.show()
