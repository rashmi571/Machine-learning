from sklearn.metrics import confusion_matrix
y_true=[0,1,1,0,1,0]#student actual data
y_pred=[1,1,0,0,0,1]#by machine predict data
confusion=confusion_matrix(y_true,y_pred)
print("confusion matrix: ",confusion)
