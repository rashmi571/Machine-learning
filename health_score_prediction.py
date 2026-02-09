#Predict patient health score based on engagement, category, donation and camp behavior.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

#Load data
df=pd.read_excel("/content/sample_data/test_data.csv.xlsx")

#clean column name
df.columns = df.columns.str.strip()

#print dta
print("\n Dataset")
print(df.head())

#drope useless data
df.drop(['Camp_Start_Date','Camp_End_Date','Unnamed: 4','Patient_ID','Health_Camp_ID','Registration_Date'],axis=1,inplace=True)

# Drop rows where 'Health Score' is NaN to handle missing target values
df_cleaned = df.dropna(subset=['Health Score']).copy()

#define target and features using the cleaned DataFrame
x = df_cleaned.drop('Health Score',axis=1) 
y = df_cleaned['Health Score']             

#find number and objected columns
catgorical_col=x.select_dtypes(include="object").columns
numberical_col=x.select_dtypes(include=np.number).columns

#handling missing value and encode
catgori=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

#handling missing values and convert into same scaler
numberical=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

#columns transfer
preprocessor=ColumnTransformer(
    transformers=[
        ('num',numberical,numberical_col),
        ('cat',catgori,catgorical_col)
    ]
)


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=10
    ))
])

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Train model
model.fit(x_train,y_train)

#prediction
y_pred=model.predict(x_test)

#Evaluation
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmsr=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

#plot
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred,color='blue',label='Prediction')
plt.plot(y_test,y_test,color='red',label='Perfect fit')
plt.xlabel("Actual Health Score")
plt.ylabel("Predicted Health Score")
plt.title("Actual vs Predicted Health Score")
plt.legend()
plt.show()

print("Mean Absolute Error: ",np.round(mae,2))
print("Mean squared Error: ",np.round(mse,2))
print("Root Mean Squared Error: ",np.round(rmsr,2))
print("R2 score: ",np.round(r2,2))

print("\n\n------------user prediction---------------")
try:
  user_data={
      'Var1':[40],
      'Var2':[2],
      'Var3':[50],
      'Var4':[10],
      'Var5':[20],
      'Online_Follower': [0],
      'LinkedIn_Shared': [0],
      'Twitter_Shared': [0],
      'Facebook_Shared': [0],
      'Income':[5],
      'Education_Score':[5],
      'Age':[30],
      'First_Interaction': ['01-Jan-00'], 
      'City_Type': ['A'],
      'Employer_Category': ['P'],
      'Category1':['second'],
      'Category2':['A'],
      'Category3':[2],
      'Donation':[90],
      'Number_of_stall_visited':[2],
      'Last_Stall_Visited_Number':[1],
  }
  #Convert dictionary to DataFrame
  user_input_df = pd.DataFrame(user_data)

  #Predict using the DataFrame
  model_predict=model.predict(user_input_df)

  print(f"Prediction of health score: {np.round(model_predict[0],2)}")

except Exception as e:
  print("Error! ",e)