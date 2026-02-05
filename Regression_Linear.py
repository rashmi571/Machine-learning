#regrestions for predict number
#1.Linear regression
#it fine the pattern and create straght line

from sklearn.linear_model import LinearRegression


x=[[1],[2],[3],[4],[5]]
y=[50,65,75,85,90]

model=LinearRegression()

model.fit(x,y)#fit is use to tranning
Hours=float(input("enter the how many hours you studies: "))

predict_marks=model.predict([[Hours]])
print("Marks predict accordig to your study hours: ",predict_marks)

