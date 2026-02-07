import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("/content/sample_data/indian_engineering_student_placement.csv")

# Drop ID column
df = df.drop('Student_ID', axis=1)

# Separate X and y first
X = df.drop(['placement_status', 'salary_lpa'], axis=1)
y = df['placement_status']

# Identify column types from X
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=np.number).columns

# Pipelines
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_pipeline, categorical_cols),
        ('num', numerical_pipeline, numerical_cols)
    ]
)

# Final Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train-test split (X and y are already defined)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Classification Report")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# -------- USER INPUT --------
print("-------- User Prediction --------")

try:
    user_data = {
        'gender': ['Male'],
        'branch': ['CSE'],
        'cgpa': [8.2],
        'tenth_percentage': [85],
        'twelfth_percentage': [82],
        'backlogs': [0],
        'study_hours_per_day': [5],
        'attendance_percentage': [90],
        'projects_completed': [3],
        'internships_completed': [1],
        'coding_skill_rating': [8],
        'communication_skill_rating': [7],
        'aptitude_skill_rating': [7],
        'hackathons_participated': [2],
        'certifications_count': [4],
        'sleep_hours': [7],
        'stress_level': [7], # Added missing 'stress_level' for user_data
        'part_time_job': ['No'],
        'family_income_level': ['Medium'], # Added missing 'family_income_level'
        'city_tier': ['Tier 1'], # Added missing 'city_tier'
        'internet_access': ['Yes'],
        'extracurricular_involvement': ['Medium'] # Added missing 'extracurricular_involvement'
    }

    user_df = pd.DataFrame(user_data)
    prediction = model.predict(user_df)

    print("Placement Status Prediction:", prediction[0])

except Exception as e:
    print("Error:", e)