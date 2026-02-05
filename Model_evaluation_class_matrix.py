#Model evaluation and matrix
#1. classification matrix

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#true answer(what actually happend)
y_true=[0,1,1,0,1,0]

#model: predicitions (what it gussed)
y_pred=[1,1,0,0,0,1]

#evaluation
print("Accurancy: ",accuracy_score(y_true,y_pred))
print("Precision: ",precision_score(y_true,y_pred))
print("Recall: ",recall_score(y_true,y_pred))
print("F1: ",f1_score(y_true,y_pred))