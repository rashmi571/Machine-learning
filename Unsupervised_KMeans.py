from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

#sample data
data={
    'customer':["rashmi","riya","muskan","diya","priya","rahul","sneha","neha"],
    'age':[20,30,40,50,90,70,22,80],
    'spending':[100,200,370,150,800,307,130,800]
}
df=pd.DataFrame(data)

#selecting the feature for clustering
x=df[["age","spending"]]

#kmeans model create that group data
# random state=that start random point
#n-init=when data stop 10n that analysis best value
model=KMeans(n_clusters=3,random_state=42,n_init=10)

#group new column
df['Group']=model.fit_predict(x)

#graph

plt.figure(figsize=(6,5))
for group in df['Group'].unique():#unique is most important [0,1]without data work unnessasary
  group_data=df[df['Group']==group]
  plt.scatter(group_data['age'],group_data['spending'],label=f'Group{group+1}')
plt.xlabel('Age')
plt.ylabel('Spending')
plt.title('Customer Clustering(k-means)')
plt.grid(True)
plt.legend()
plt.show()

print(df)