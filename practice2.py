#practice question
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
data={
    "studyHours":[1,2,3,4,5],
    "marks":[10, 12, 14, 16, 18]
}
df=pd.DataFrame(data)
minmax_scaler=MinMaxScaler()
minmax_scaled=minmax_scaler.fit_transform(df)

print("Minmax scaler")
print(pd.DataFrame(minmax_scaled,columns=["studyHours","marks"]))