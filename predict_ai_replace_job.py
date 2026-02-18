import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

#data loading as csv file
df=pd.read_csv("/content/sample_data/ai_job_replacement_industry_nan_dataset.csv")

# Drop rows where the target variable is NaN
df.dropna(subset=['AI_Replace_Job'], inplace=True)

x=df.drop(['AI_Replace_Job'], axis=1) 
y=df['AI_Replace_Job']

#classifie number and catgori columns
catgorical_col=x.select_dtypes(include='object').columns
numberical_col=x.select_dtypes(include=np.number).columns

#using pipeline 
catgori=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')), 
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

number=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

#using columnstransformer
preprocessor=ColumnTransformer(
    transformers=[
        ('cat',catgori,catgorical_col),
        ('num',number,numberical_col)
    ]
)

model=Pipeline(steps=[
    ('preprocessor',preprocessor),
     ('classifier', LogisticRegression(max_iter=1000))
])

#divide data for test and train
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,random_state=42) 

#model train
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("confusion matrix")
conf_m=confusion_matrix(y_test,y_pred)
print(conf_m)

print("Accuracy:")
print(accuracy_score(y_test, y_pred))


# Visualization
sea.heatmap(conf_m, annot=True, fmt="d",cmap='Blues') 
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

try:
    user_data=pd.DataFrame({
        'Age':[20],
        'Education_Level':['Graduate'], 
        'Experience_Years':[5], 
        'Industry':['IT'],
        'Job_Role':['Data Analyst'],
        'Technical_Skills_Score':[5],
        'Creativity_Score':[6],
        'Communication_Score':[6],
        'AI_Knowledge':[1],
        'Automation_Risk':[5]
    })

    prediction=model.predict(user_data)

    print(f'Prediction using userdata: ',prediction[0])

except Exception as e:
    print("Error!",e)