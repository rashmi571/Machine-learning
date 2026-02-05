from sklearn.preprocessing import LabelEncoder
import pandas as pd
df=pd.read_csv("/content/sample_data/sample_txt.csv")

#for label encoding
df_label=df.copy()
le=LabelEncoder()
df_label['gender_encoder'] = le.fit_transform(df['gender'])
df_label['city_encoder'] = le.fit_transform(df['city'])

print("\n Label encoded data")
print(df_label[["name","gender","city","passed","gender_encoder","city_encoder"]])

#for one hot encoding
df_en=pd.get_dummies(df,columns=['city'],dtype=int)
print("\n One hot encoded data")
print(df_en)