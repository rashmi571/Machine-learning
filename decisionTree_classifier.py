#decisiontree
from sklearn.tree import DecisionTreeClassifier
x=[
    [7,2],
    [8,3],
    [9,8],
    [10,9]
    ]
y=[0,0,1,1]#0=Apple,1=orange

model=DecisionTreeClassifier()
model.fit(x,y)

size=float(input("enter the size of fruit: "))
shade=float(input("enter the shade of fruit: "))

predict_fruit=model.predict([[size,shade]])[0]

if predict_fruit == 0:
  print("Apple")
else:
  print("Orange")

