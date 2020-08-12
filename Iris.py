import numpy as np
from sklearn import datasets
#load iris dataset
iris=datasets.load_iris()
X=iris.data
#find the number of classes 
numberClass=np.array(list(set(iris.target))).shape[0]
#use for convert dataset array [class1,class1,...,class2,class2,...,classN,classN..] to [class1,class2,class3...classN,...,class1,class2,class3...classN,...] to train easily.
def balance(array,numberClass):
 new_array=[]
 end=int(array.shape[0]/numberClass)
 for i in range(end):
  for j in range(numberClass):
   new_array.append(array[i+end*j])
 return np.array(new_array)
X=balance(X,numberClass)
Y=np.zeros((iris.target.shape[0],numberClass))
for i in range(X.shape[0]):
 Y[i,i%numberClass]=1
stringNumtrain="Size max is "+str(X.shape[0])+".The number of elements for training is "
Numtrain=int(input(stringNumtrain))
X_train=X[:Numtrain]
Y_train=Y[:Numtrain]
X_test=X[Numtrain:]
Y_test=Y[Numtrain:]
#create parameters
weight=np.random.rand(X.shape[1],numberClass)
bias=np.zeros((1,numberClass))
learning_rate=0.01
stringTrainLoop="The number of iteration is "
train_loop=int(input(stringTrainLoop))
#softmax function
def softmax(Z):
 eZ=Z
 for i in range(Z.shape[0]):
  eZ[i]=np.exp(eZ[i]-np.max(eZ[i]))/np.sum(np.exp(eZ[i]-np.max(eZ[i])))
 return eZ
#the loop for training with Softmax Regression algorithm
for i in range(train_loop):
 Y_predict=softmax(X_train.dot(weight)+bias)
 weight-=learning_rate*X_train.T.dot(Y_predict-Y_train)
 bias-=learning_rate*np.sum(Y_predict-Y_train,axis=0)
#test 
test_false=0
Y_predict=softmax(X_test.dot(weight)+bias)
for i in range(Y_test.shape[0]):
 if(Y_test[i,i%numberClass]==1 and Y_predict[i,i%numberClass] < 0.5):
  test_false+=1
score=(Y_test.shape[0]-test_false)/Y_test.shape[0]
stringScore="Accuracy: "+str(score*100.)+"%."
print(stringScore)