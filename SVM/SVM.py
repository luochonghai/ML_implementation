# coding = <utf-8>

'''SVM for classification on breast_cancer data set'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
'''the sklearn.metrics model includes score functions,
performance metrics and pairwise metrics and distance computations'''
from sklearn import metrics
from sklearn import svm

'''loading data'''
data_set = load_breast_cancer()

X = data_set.data #feature
feature_names = data_set.feature_names
y = data_set.target #label
target_names = data_set.target_names

f1 = plt.figure(1)
p1 = plt.scatter(X[y == 0,0],X[y == 0,1],color = 'r',label = target_names[0]) #feature
p2 = plt.scatter(X[y == 1,0],X[y == 1,1],color = 'g',label = target_names[1]) #feature
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend(loc = 'upper right')
plt.grid(True,linewidth = 0.3)

plt.show()

'''data normalization'''
normalized_X = preprocessing.normalize(X)

'''generation of train set and testing set'''
X_train,X_test,y_train,y_test = train_test_split(normalized_X,y,test_size = 0.5,random_state = 0)

'''model fitting,testing,visualization
based on linear kernel and rbf model'''
for fig_num,kernel in enumerate(('linear','rbf','poly','sigmoid')):#enumerate func adds index to sequence
	accuracy = []
	c = []
	for C in range(1,10000,1):#the third parameter is the pace
		clf = svm.SVC(C = C,kernel = kernel)#C means penalty parameter of the error term
		clf.fit(X_train,y_train)#train
		y_pred = clf.predict(X_test)#test
		accuracy.append(metrics.accuracy_score(y_test,y_pred))
		c.append(C)

	print('max accuracy of %s kernel SVM: %.4f'%(kernel,max(accuracy)))

	'''draw accuracy'''
	f2 = plt.figure(2)
	plt.plot(c,accuracy)
	plt.xlabel('penalty parameter')
	plt.ylabel('accuracy')
	plt.show()
