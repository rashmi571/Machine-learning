import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.decomposition import PCA

#data
df=pd.read_csv("/content/sample_data/kmeans_ai_job_dataset_8000_rows.csv")

#saperate columns
catgori_col=df.select_dtypes(include='object').columns
number_col=df.select_dtypes(include=np.number).columns

#pipeline
number=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

catgori=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

#columns transfer
preprocessor=ColumnTransformer(
    transformers=[
        ('num',number,number_col),
        ('cat',catgori,catgori_col)
    ]
)


#elbow method
processed_data=preprocessor.fit_transform(df)

inertia=[]

for k in range(1,10):
    km=KMeans(n_clusters=k,random_state=42)
    km.fit(processed_data)
    inertia.append(km.inertia_)


plt.plot(range(1,10),inertia)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow method")
plt.show()

#apply Kmeans(choose k= 3)
Kmeans = KMeans(n_clusters=3,random_state=42)
clusters=Kmeans.fit_predict(processed_data)

#add clusters to data frame
df['cluster']=clusters

print(df.head())

#For using PCA
pca=PCA(n_components=2)
reduce_data=pca.fit_transform(processed_data)

#chart
plt.scatter(reduce_data[:, 0], reduce_data[:, 1], c=clusters)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KMeans Clustering (PCA View)")
plt.show()