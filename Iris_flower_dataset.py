import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


df=pd.read_csv("/content/sample_data/IRIS.csv")

x=df.drop(['species'],axis=1)
y=df['species']

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

k=KNeighborsClassifier(n_neighbors=10)
k.fit(x_train, y_train) # Fit the model on the training data

y_pred=k.predict(x_test) # Make predictions on the test features

print("Classifiction report")
print(classification_report(y_test,y_pred))

print('\nconfusion matrix')
print(confusion_matrix(y_test,y_pred))

print("Accurany Score: ",accuracy_score(y_test,y_pred))

user_input=pd.DataFrame({
    'sepal_length':[6.0],
    'sepal_width':[3.7],
    'petal_length':[1.6],
    'petal_width':[0.4]
})

user_prediction=k.predict(user_input)

print('Prediction user input: ',user_prediction[0])