import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns

df=pd.read_csv("/content/sample_data/student_performance_data.csv")

le=LabelEncoder()
df['internet_use']=le.fit_transform(df['internet_use'])
df['passed']=le.fit_transform(df['passed'])

features=['studyhour','internet_use','pastscore','attendance']

scaler=StandardScaler()
df_scaled=df.copy()
df_scaled[features]=scaler.fit_transform(df[features])

x=df_scaled[features]
y=df['passed'] 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

predict_model=model.predict(x_test)

print("Classification Report")
print(classification_report(y_test,predict_model))

conf_matrix=confusion_matrix(y_test,predict_model)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',cbar=False,xticklabels=['Fail','Pass'],yticklabels=['Fail','Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("---------------predict your result---------------")
try:
  Studyhours=float(input("Enter your study hours: "))
  Internet_use=int(input("enter your internet use: "))
  Pastscore=float(input("enter your past score: "))
  Attendance=float(input("enter your attendance: "))

  user_input_df=pd.DataFrame([{
      'studyhour':Studyhours,
      'internet_use':Internet_use,
      'pastscore':Pastscore,
      'attendance':Attendance
  }])

  user_input_scaled=scaler.transform(user_input_df)
  user_predict=model.predict(user_input_scaled)

  result='Pass' if user_predict[0] == 1 else 'Fail'
  print(f"Prediction based input: {result}")
except Exception as e:
  print("An error occured",e)