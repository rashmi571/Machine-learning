#for practice questions
from sklearn.preprocessing import LabelEncoder
import pandas as pd
df=pd.read_csv("/content/sample_data/sample_txt.csv")

df_label=df.copy()
le=LabelEncoder()
df_label['gender_encoder'] = le.fit_transform(df['gender'])

print(df_label[["name","city","passed","gender_encoder"]])

df_encode=pd.get_dummies(df_label,columns=['city'],dtype=int)
print(df_encode)
