import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.linear_model import LinearRegression # Changed to LinearRegression for regression task
from sklearn.impute import SimpleImputer

df=pd.read_csv("/content/sample_data/indian_engineering_student_placement.csv")

df=df.drop('Student_ID',axis=1)

x=df.drop('salary_lpa',axis=1)
y=df['salary_lpa']


categorical_col=x.select_dtypes(include='object').columns
numerical_col=x.select_dtypes(include=np.number).columns

categori=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

numerical=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

preprocessor=ColumnTransformer(
    transformers=[
        ('num',numerical,numerical_col),
        ('cat',categori,categorical_col)
    ])
model=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',LinearRegression())
])


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print("--------User Prediction--------")
try:
  user_data={
      'gender':['Male'],
      'branch':['CSE'],
      'cgpa':[8.2],
      'tenth_percentage':[85],
      'twelfth_percentage':[82],
      'backlogs':[0],
      'study_hours_per_day':[5],
      'attendance_percentage':[90],
      'projects_completed':[3],
      'internships_completed':[1],
      'coding_skill_rating': [8],
        'communication_skill_rating': [7],
        'aptitude_skill_rating': [7],
        'hackathons_participated': [2],
        'certifications_count': [4],
        'sleep_hours': [7],
        'stress_level': [7],
        'part_time_job': ['No'],
        'family_income_level': ['Medium'],
        'city_tier': ['Tier 1'],
        'internet_access': ['Yes'],
        'extracurricular_involvement': ['Medium'],
        'placement_status': ['Placed']
  }
  user_pred=pd.DataFrame(user_data)
  prediction=model.predict(user_pred)

  print("Salary Prediction:",np.round(prediction[0],2))

except Exception as e:
  print("Error:",e)