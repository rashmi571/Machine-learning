#how far my model
#Mean absolute error
#Mean squared error
#Root mse

from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np

#real value
real_value=[90,60,80,100]

#predict by model
predict_value=[95,70,70,95]

mean_error=mean_absolute_error(real_value,predict_value)
print("mean absolute error: ",mean_error)

squared_error=mean_squared_error(real_value,predict_value)
print("mean squared error: ",squared_error)

rmse=np.sqrt(squared_error)
print("root mean squared error: ",rmse)

print()