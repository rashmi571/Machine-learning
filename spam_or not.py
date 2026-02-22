import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv("/content/sample_data/spam.csv", encoding='latin1')

delete=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)

#rename columns
df.columns=['label','message']

x=df['message']
y=df['label']

vectorized=TfidfVectorizer()
x_tranform=vectorized.fit_transform(x)



x_train, y_train, x_test, y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
)

model=LinearRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("Accurany: ",accuracy_score(y_test,y_pred))

#user input-------------------------------------
user_input=['you win 100000 money if you click this link']

user_tranform=vectorized.tranforme(user_input)

user_prediction=model.predict(user_tranform)

if user_prediction[0] == 'ham':
    print("msg not sapm")
else:
    print("msg is spam")    
    





