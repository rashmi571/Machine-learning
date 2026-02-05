from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data={
    'x':[10, 20, 30, 40]
}
df=pd.DataFrame(data)

min_max=MinMaxScaler()
max_scaled=min_max.fit_transform(df)

print(pd.DataFrame(max_scaled,columns=['x']))