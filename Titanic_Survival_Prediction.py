import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,precision_score,recall_score

df=pd.read_csv("")

#remove useless columns
column_TO_drop=['PassangerId','Cabin']
df=df.drop(columns=[col for col in column_TO_drop if col in df.columns])

x=df.drop(['Survived'],axis=1)
y=df['Survived']

catgori_col=x.select_dtypes(include='object').column
numbercal_col=x.select_dtypes(include=np.number).column

#using pipeline for catgori 
catgori=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequant')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

#pipeline use for numberr columns
number=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

#uding columntransfer for all
preprocessor=ColumnTransformer(
    transformers=[
        ('cat',catgori,catgori_col),
        ('num',number,numbercal_col)
    ]
)

model=Pipeline(steps=[
    ('prprocessor',preprocessor),
    ('regressor',LogisticRegression(max_iter=1000))
])


x_train, y_train, x_test, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

conf_mat=confusion_matrix(y_test,y_pred)
plt.heatmap(conf_mat,annot=True,fmt='d',cmap='blue',xtricklabels=['Not Survived','Survived'],ytricklabels=['Not Survived','Survived'])
plt.xlabel("Prediction")
plt.ylabel("Actal preediction")
plt.title("Confusion matrix")
plt.show()

a=df['Survived'].value_count()
b=df['Survived'].value_count().index

plt.pie(a,labels=b,colors='blue',autopct='%1.1f%%')
plt.title("Survived Destirbution")
plt.show()

print("classification report: ",classification_report(y_test,y_pred))

print("Accurancy Score: ",accuracy_score(y_test,y_pred))
print("Precision score: ",precision_score(y_test,y_pred,average='weighted'))
print("Recall score: ",recall_score(y_test,y_pred,average='weighted'))
print("F1 score: ",f1_score(y_test,y_pred,average='weighted'))



