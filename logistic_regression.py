#2.LogisticRegression is use to predict yes or not means binary form 0 or 1
from sklearn.linear_model import LogisticRegression
x=[[1],[2],[3],[4],[5]]
y=[0,0,1,1,1]

model=LogisticRegression()

model.fit(x,y)
hours=float(input("enter the how many hours you studies: "))
predict_marks=model.predict([[hours]])[0]

if predict_marks==0:
  print("student will fail")
else:
  print("student will pass")

