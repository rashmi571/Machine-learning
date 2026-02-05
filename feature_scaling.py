
#feature scalering

import pandas as pd
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
from sklearn.model_selection import train_test_split

data={
    'studyHours':[2,3,4,5,10,11],
    'textScore':[40,50,60,70,80,90] #give neg number because high diff in study hour or test score
}
df=pd.DataFrame(data)

#standardScaler
standard_scaler=StandardScaler()
standerd_score=standard_scaler.fit_transform(df)

print("standerd scaler")
print(pd.DataFrame(standerd_score,columns=['studyHours','textScore']))

minmax_scler=MinMaxScaler()
minmax_score=minmax_scler.fit_transform(df)

print("\nMinMax scaler")
print(pd.DataFrame(minmax_score,columns=['studyHours','textScore']))

#train test split
x=df[['studyHours']]#input
y=df[['textScore']]#output

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print("\n Traning data")
print(X_train)

print("\n Testing data")
print(X_test)

print("\n Traning data")
print(y_train)

print("\n Testing data")
print(y_test)

