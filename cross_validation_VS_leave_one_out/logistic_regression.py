# -*- coding: utf-8 -*

import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import random
from sklearn import *
import self_Logistic_Regression

'''randomize the index of data&label'''
def randomize_data(data,label):
	#numpy.nparray is not allowed to use median_variable to swap two rows or columns;you should use its special grammar
	data_dim = len(data)
	for data_iter in range(data_dim):
		temp_swap = random.randint(data_iter,data_dim-1)
		data[[data_iter,temp_swap],:] = data[[temp_swap,data_iter],:]
		label[[data_iter,temp_swap]] = label[[temp_swap,data_iter]]

	return data,label

'''two ways to import data'''
def import_data(file_path):
	dataset = np.loadtxt(file_path,delimiter = ',')
	X = dataset[:,1:3]
	y = dataset[:,3]
	#print(X)
	#print(y)
	list_file = []
	with open(file_path,'r') as csv_file:
		all_lines = csv.reader(csv_file)
		for one_line in all_lines:
			list_file.append(one_line)
	arr_file = np.array(list_file,dtype = np.float64)
	label = arr_file[:,3]
	data = arr_file[:,1:-1]

	data_cooked = randomize_data(data,label)
	data = data_cooked[0]
	label = data_cooked[1]

	#print(data)
	#print(label)

	'''draw scatter_pic'''
	f1 = plt.figure(1)
	plt.title('logistic_regression')
	plt.xlabel('density')
	plt.ylabel('ratio_sugar')
	plt.scatter(data[label == 0,0],data[label == 0,1],marker = 'o',color = 'r',s = 100,label = 'bad')
	plt.scatter(data[label == 1,0],data[label == 1,1],marker = 'o',color = 'g',s = 100,label = 'good')
	plt.legend(loc = 'upper left') 
	plt.show()

	return data,label

'''use sklearn lib for logistic regression'''
def sklearn_logistic_regression(X,y):
	X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.5,random_state = 0)

	log_model = linear_model.LogisticRegression()
	log_model.fit(X_train,y_train)

	y_pred = log_model.predict(X_test)

	print(metrics.confusion_matrix(y_test,y_pred))
	print(metrics.classification_report(y_test,y_pred))

	precision,recall,thresholds = metrics.precision_recall_curve(y_test,y_pred)

	f2 = plt.figure(2)
	h = 0.001
	delta = 0.1#used to restrict the space for pic
	x0_min,x0_max = X[:,0].min()-delta,X[:,0].max()+delta
	x1_min,x1_max = X[:,1].min()-delta,X[:,1].max()+delta
	x0,x1 = np.meshgrid(np.arange(x0_min,x0_max,h),np.arange(x1_min,x1_max,h))

	z = log_model.predict(np.c_[x0.ravel(),x1.ravel()])

	z = z.reshape(x0.shape)
	plt.contourf(x0,x1,z,cmap = 'viridis')# pl.cm.Paired

	plt.title('watermelon_3a')
	plt.xlabel('density')
	plt.ylabel('ratio_sugar')
	plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'o', color = 'k', s=100, label = 'bad')
	plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=100, label = 'good')
	plt.show()

def self_LR(X,y):
	m,n = np.shape(X)
	X_ex = np.c_[X,np.ones(m)]#extend the variable matrix to x^0
	X_train,X_test,y_train,y_test = model_selection.train_test_split(X_ex,y,test_size = 0.5,random_state = 0)

	#using gradDescent to get the optimal param beta = [w,b]
	beta = self_Logistic_Regression.gradDscent_1(X_train,y_train)

	#prediction,beta mapping to the model
	y_pred = self_Logistic_Regression.predict(X_test,beta)

	m_test = np.shape(X_test)[0]
	#calculation of confusion_matrix and prediction accuracy
	cfmat = np.zeros((2,2))
	for i in range(m_test):
		if y_pred[i] == y_test[i] and y_test[i] == 0:
			cfmat[0,0] += 1
		elif y_pred[i] == y_test[i] and y_test[i] == 1:
			cfmat[1,1] += 1
		elif y_pred[i] == 0:
			cfmat[1,0] += 1
		elif y_pred[i] == 1:
			cfmat[0,1] += 1
	print(cfmat)


if __name__ == '__main__':
	dataset = import_data('D:\\FDU\Template\\CS\\Machine Learning\\周志华西瓜书编程练习\\ch3线性模型\\3.3_logistic_regression\\watermelon_3a.csv')
	self_LR(dataset[0],dataset[1])
	sklearn_logistic_regression(dataset[0],dataset[1])