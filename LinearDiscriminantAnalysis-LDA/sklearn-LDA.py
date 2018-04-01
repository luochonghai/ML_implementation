# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import *
from sklearn import *
from self_LDA import *

def LDA_sklearn(X,y):
	X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.5,random_state = 0)
	lda_model = discriminant_analysis.LinearDiscriminantAnalysis(solver = 'lsqr',shrinkage = None).fit(X_train,y_train)
	y_pred = lda_model.predict(X_test)

	print(metrics.confusion_matrix(y_test,y_pred))
	print(metrics.classification_report(y_test,y_pred))

	f2 = plt.figure(1)
	h = 0.001
	x0_min,x0_max = X[:,0].min()-0.1,X[:,0].max()+0.1
	x1_min,x1_max = X[:,1].min()-0.1,X[:,0].max()+0.1

	x0,x1 = np.meshgrid(np.arange(-1,1,h),np.arange(x1_min,x1_max,h))

	z = lda_model.predict(np.c_[x0.ravel(),x1.ravel()])

	z = z.reshape(x0.shape)
	plt.contourf(x0,x1,z)

	plt.title('watermelon_3a')
	plt.xlabel('density')
	plt.ylabel('ratio_sugar')
	plt.scatter(X[y == 0,0],X[y == 0,1],marker = 'o',color = 'k',s = 100,label = 'bad')
	plt.scatter(X[y == 1,0],X[y == 1,1],marker = 'o',color = 'g',s = 100,label = 'good')
	plt.show()

if __name__ == '__main__':
	#file_path = 'D:\\FDU\Template\\CS\\Machine Learning\\周志华西瓜书编程练习\\ch3线性模型\\3.3_logistic_regression\\watermelon_3a(no_outlier).csv'
	file_path = 'D:\\FDU\Template\\CS\\Machine Learning\\周志华西瓜书编程练习\\ch3线性模型\\3.3_logistic_regression\\watermelon_3a.csv'
	dataset = import_data(file_path)
	#LDA_sklearn(dataset[0],dataset[1])
	LDA_self_imp(dataset[0],dataset[1])
