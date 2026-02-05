#PCA(principle component ananlysis)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data={
    'Age':[25,30,35,40,45,50],
    'Income':[3000,4000,50000,60000,70000,80000],
    'Spendings':[70,60,50,40,30,20],
    'Savings':[1000,5000,8000,10000,15000,20000]
}
df=pd.DataFrame(data)

#standardize the data
scaler=StandardScaler()
scaled_data=scaler.fit_transform(df)

#pca
pca=PCA(n_components=2)#4 columns exist in data -> only 2 column converrt
pca_data=pca.fit_transform(scaled_data)

#create new columns
pca_df=pd.DataFrame(data=pca_data,columns=['PCA1','PCA2'])

#how much data use by using explained variance ratio
explained_variance=pca.explained_variance_ratio_
print("Vaariance caputure by each PCA components")
print(np.round(explained_variance * 100,2))#answer in %

#plot
plt.figure(figsize=(8,6))
plt.scatter(pca_df['PCA1'],pca_df['PCA2'],color='blue',s=80)
plt.xlabel('PCA1 main pattern')
plt.ylabel('PCA2 minor pattern')
plt.title('PCA Analysis(2d view)')
plt.grid(True)
plt.show()

print("New data with 2 feature PCA1  pCA2")
print(pca_df)