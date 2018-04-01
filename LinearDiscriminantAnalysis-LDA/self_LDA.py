# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import *
from sklearn import *
from pjpoint import *

def LDA_self_imp(X,y):
	'''step1:to get the mean vector of each class'''
	u = []
	for i in range(2):#just 2 class
		u.append(np.mean(X[y == i],axis = 0))

	'''step2:compute the within-class scatter matrix,referenced to book(3.33)'''
	m,n = np.shape(X)
	Sw = np.zeros((n,n))
	for i in range(m):
		x_tmp = X[i].reshape(n,1)
		if y[i] == 0:
			u_tmp = u[0].reshape(n,1)
		else:
			u_tmp = u[1].reshape(n,1)
		Sw+= np.dot(x_tmp-u_tmp,(x_tmp-u_tmp).T)

	Sw = np.mat(Sw)#now you can modify Sw's element 
	U,sigma,V = np.linalg.svd(Sw)

	Sw_inv = V.T*np.linalg.inv(np.diag(sigma))*U.T

	'''step3:compute the parameter w,referenced to book(3.39)'''
	w = np.dot(Sw_inv,(u[0]-u[1]).reshape(n,1))
	print(w)

	'''step4:draw the LDA line in scatter figure'''
	f2 = plt.figure(2)
	plt.xlim(-0.2,1)
	plt.ylim(-0.5,0.7)

	p0_x0 = -X[:,0].max()
	p0_x1 = (w[1,0]/w[0,0])*p0_x0
	p1_x0 = X[:,0].max()
	p1_x1 = (w[1,0]/w[0,0])*p1_x0

	plt.title('watermelon_3a-self_LDA')
	plt.xlabel('density')
	plt.ylabel('ratio_sugar')
	plt.scatter(X[y == 0,0],X[y == 0,1],marker = 'o',color = 'k',s = 10,label = 'bad')
	plt.scatter(X[y == 1,0],X[y == 1,1],marker = 'o',color = 'g',s = 10,label = 'good')
	plt.legend(loc = 'upper right')

	plt.plot([p0_x0,p1_x0],[p0_x1,p1_x1])

	#draw projective point on the line
	m,n = np.shape(X)
	for i in range(m):
		x_p = GetProjectivePoint([X[i,0],X[i,1]],[w[1,0]/w[0,0],0])
		if y[i] == 0:
			plt.plot(x_p[0],x_p[1],'ko',markersize = 5)
		else:
			plt.plot(x_p[0],x_p[1],'go',markersize = 5)
		plt.plot([x_p[0],X[i,0]],[x_p[1],X[i,1]],'c--',linewidth = 0.3)
	plt.show()

