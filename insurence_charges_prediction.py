import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score 

#load data
df=pd.read_csv("/content/sample_data/insurance.csv")

x=df.drop(['charges'],axis=1)
y=df['charges']

#describe the numberical and categorical columns
catgori_col=x.select_dtypes(include='object').columns 
number_col=x.select_dtypes(include=np.number).columns 


#fill nan valuse and same scales
number=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
    ])

#fill nan catgoris and encode
catgori=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')), 
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

#transform all columns
preprocessor=ColumnTransformer(
    transformers=[
        ('num',number,number_col),
        ('cat',catgori,catgori_col)
    ]
)

#manage for prediction by using randomforest
model=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('regressor',RandomForestRegressor( # Corrected 'regrossor' to 'regressor'
         n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1

    ))
])

#desribe the data for test and train
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42) # Corrected unpacking order

#give data for training
model.fit(x_train,y_train)

#give data for prdict
y_pred=model.predict(x_test)

#evaluation
mae=mean_absolute_error(y_test,y_pred) # Corrected typo
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("Mean absolute error: ",np.round(mae,2))
print("Mean squared error: ",np.round(mse,2))
print("Root mean squared error: ",np.round(rmse,2))
print("R2 Score: ",np.round(r2,2))

#input for user 
try:
    user_data=pd.DataFrame({
        'age':[50],
        'sex':['male'],
        'bmi':[45.9],
        'children':[4], # Corrected 'childern' to 'children'
        'smoker':['yes'],
        'region':['southwest']
    })

    #user input using for prediction
    model_prediction=model.predict(user_data)

    #print user prediction by model
    print(f"Prediction of user data: {np.round(model_prediction,2)[0]}")

#if any error using try and except
except Exception as e:
    print('Error!',e)
