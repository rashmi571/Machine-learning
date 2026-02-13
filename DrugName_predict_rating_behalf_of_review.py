import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv("/content/sample_data/obat.csv")

df=df.drop(['uniqueID','date'],axis=1)

catgorical_col=['drugName','condition']
text_col='review' # Changed this from list to string
numberical_col = ['usefulCount']

catgori = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

number = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

text = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=5000))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', catgori, catgorical_col),
        ('num', number, numberical_col),
        ('text', text, text_col)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    ))
])


x=df.drop(['rating'],axis=1)
y=df['rating']

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmsr=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred,color='blue',label='Prediction')
plt.plot(y_test,y_test,color='red',label='Perfect fit')
plt.xlabel("Actual rating")
plt.ylabel("Predicted rating")
plt.title("Actual vs Predicted rating ")
plt.legend()
plt.show()

print("Mean Absolute Error: ",np.round(mae,2))
print("Mean squared Error: ",np.round(mse,2))
print("Root Mean Squared Error: ",np.round(rmsr,2))
print("R2 score: ",np.round(r2,2))

print("\n\n------------user prediction---------------")
try:
  user_data=pd.DataFrame({
      'drugName':['Orthovisc'],
      'condition':['headache'],
      'review':["What a waste of money I paid a copay of $400 for 3 injections in my left and right knee. I am bone on bone they are worst now after the shots ortho visc should be investigated for false claims on what their product doesn&#039;t do, very dissapointed. I tryed to complain on their site but there is no contact us on their page"],
      'usefulCount':[1]
  })
  model_predict=model.predict(user_data)

  print(f"Prediction of rating: {np.round(model_predict[0],2)}")
except Exception as e:
  print("Error!",e)
  


