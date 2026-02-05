#knighbors is use to predict data accordint to neighbors decision
#take a example of fruit
#weight(gm),Size(cm),which fruit predict
from sklearn.neighbors import KNeighborsClassifier
x=[
    [180,7],
    [200,7.5],
    [250,8],
    [300,8.5],
    [330,9],
    [360,9.5]
]
#0=Apple,1=Orange
y=[0,0,0,1,1,1]

model=KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)

weight=float(input("enter the weight of fruit: "))
size=float(input("enter the size of fruit: "))

predict_fruit=model.predict([[weight,size]])[0]#only write contain value with [] this box

if predict_fruit == 0:
  print("Apple")
else:
  print("Orange")