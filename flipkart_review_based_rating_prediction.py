import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer 

df=pd.read_csv("/content/sample_data/data.csv")

text_vectorizer = TfidfVectorizer() 
text_transformed = text_vectorizer.fit_transform(df['review']) 

#elbow method
inertia=[]

for k in range(1,10):
    km=KMeans(n_clusters=k,random_state=42, n_init=10) #
    km.fit(text_transformed) # Fit on the transformed text data
    inertia.append(km.inertia_)

plt.plot(range(1,10),inertia)
plt.xlabel("Number of clusters(k)")
plt.ylabel("Inertia")
plt.title("elbow method")
plt.show()

#apply Kmeans
Kmean=KMeans(n_clusters=3,random_state=42, n_init=10) 
clusters=Kmean.fit_predict(text_transformed) # Fit and predict on the transformed text data

#add cluster as dataframe
df['cluster']=clusters

print(df.head())


user_input=['product is very bad']


user_transform=text_vectorizer.transform(user_input)

predict_cluster=Kmean.predict(user_transform)

print("user input prediction rating : ",predict_cluster[0])

user_cluster=Kmean.predict(user_transform)