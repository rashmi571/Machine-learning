import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

#data loading
df=pd.read_csv("/content/sample_data/imdb_movies_shows.csv")

#print preview of data
print(df.head())

#drop useless columns
column_drop=['release_year','imdb_id']
df.drop(columns=[ col for col in column_drop if col in df.columns],inplace=True)

# Handle NaN values in the target column 'imdb_score'
df.dropna(subset=['imdb_score'], inplace=True)

#set target
x=df.drop(['imdb_score'],axis=1)
y=df['imdb_score']

#describe the catgori and numberical columns
catgorical_col=x.select_dtypes(include='object').columns
numberical_col=x.select_dtypes(include=np.number).columns

#using pipeline
catgori=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

number=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scale',StandardScaler())
])

#columntransfer
preprocessor=ColumnTransformer(
    transformers=[
        ('num',number,numberical_col),
        ('cat',catgori,catgorical_col)
    ]
)

#using pipeline for model
model=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('regressor',RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42
    ))
])

#model training
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,random_state=42)

model.fit(x_train,y_train)

#model for testing
y_pred=model.predict(x_test)

#model evaluation
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("Mean absolute error: ",np.round(mae,2))
print("Mean squared error: ",np.round(mse,2))
print("Root mean squared error: ",np.round(rmse,2))
print("R2 Score: ",np.round(r2,2))

#data visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted IMDb Score")
plt.show()


#user input
try:
    user_data=pd.DataFrame({
        'title':['Five Came Back: The Reference Films'],
        'type':['SHOW'],
        'age_certification':['R'],
        'runtime':[100],
        'genres':['horror'],
        'production_countries':['GB'],
        'seasons':[3],
        'imdb_votes':[10112]

    })

    prediction=model.predict(user_data)

    print("Prediction by using user input: ",np.round(prediction,2)[0])

except Exception as e:
    print("Error!",e)