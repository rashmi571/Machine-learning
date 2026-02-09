#Google Play Store – Most Downloaded Android Apps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("/content/sample_data/google_play_store_most_downloaded_apps.csv")

X = df.drop(['App', 'Category'], axis=1)
y = df['Category']

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=np.number).columns

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pred_counts = pd.Series(y_pred).value_counts()

plt.figure(figsize=(8,6))
plt.pie(pred_counts.values,
        labels=pred_counts.index,
        autopct='%1.1f%%',
        startangle=90)
plt.title("Predicted App Category Distribution")
plt.show()

user_data = pd.DataFrame({
    'Developer': ['Google'],
    'Downloads': ['10B+'],
    'Date_Reached': ['2023-07-01'],
    'Date_Published': ['2012-09-26'],
    'Pre_installed': ['Yes'],
    'Type': ['Free'],
    'Price': [0]
})

print(model.predict(user_data)[0])
