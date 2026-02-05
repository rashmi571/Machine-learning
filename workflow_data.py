# Loan Default Prediction

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Import SimpleImputer

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("/content/sample_data/Loan_Default.csv")

# -------------------------------
# 2. Target & Feature Separation
# -------------------------------
y = df["Status"]          # target
X = df.drop(columns=["Status", "ID", "year"])  # drop useless columns

# -------------------------------
# 3. Identify column types
# -------------------------------
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(include=np.number).columns

# -------------------------------
# 4. Preprocessing
# -------------------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Impute numerical NaNs with the mean
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute categorical NaNs with the most frequent value
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# -------------------------------
# 5. Model Pipeline
# -------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# -------------------------------
# 6. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 7. Train Model
# -------------------------------
model.fit(X_train, y_train)

# -------------------------------
# 8. Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 9. Evaluation
# -------------------------------
print("Classification Report\n")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

# -------------------------------
# 10. Confusion Matrix Plot
# -------------------------------
plt.figure(figsize=(7,5))
sns.heatmap(conf_matrix, annot=True, fmt="d",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# 11. User Prediction
# -------------------------------
try:
    user_data = {
        "loan_limit": input("loan_limit: "),
        "Gender": input("Gender: "),
        "approv_in_adv": input("approv_in_adv: "),
        "loan_type": input("loan_type: "),
        "loan_purpose": input("loan_purpose: "),
        "Credit_Worthiness": input("Credit_Worthiness: "),
        "open_credit": input("open_credit: "),
        "business_or_commercial": input("business_or_commercial: "),
        "loan_amount": float(input("loan_amount: ")),
        "income": float(input("income: ")),
        "Credit_Score": int(input("Credit_Score: ")),
        "Region": input("Region: ")
    }

    user_df = pd.DataFrame([user_data])
    prediction = model.predict(user_df)

    print("\nLoan Default Prediction:",
          "Default" if prediction[0] == 1 else "No Default")

except Exception as e:
    print("Error:", e)
