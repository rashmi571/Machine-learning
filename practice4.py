from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

data={
    "hours":[2,3,4,5,6,7,8,9],
    "score":[30,40,50,60,70,80,90,100]
}
df=pd.DataFrame(data)

x=df[["hours"]]
y=df["score"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=KNeighborsRegressor(n_neighbors=3)
model.fit(x_train,y_train)

predict_score=model.predict(x_test)

new_hour=int(input("enter the the new hour: "))
new_hours=pd.DataFrame([[new_hour]],columns=x.columns)

# Corrected line: pass the DataFrame directly
new_predict_score=model.predict(new_hours)
print(f"predict for {new_hours} hour: {new_predict_score[0]}")