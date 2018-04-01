# -*- coding: utf-8 -*

#object likelihood function
import numpy as np
import math
import matplotlib.pyplot as plt

def likelihood_sub(x,y,beta):
	'''
	@param X: one sample variables
	@param y: one sample label
	@param beta: the parameter vector in 3.27
	@return: the sub_log-likelihood of 3.27
	'''

	return -y * np.dot(beta,x.T)+np.math.log(1+np.exp(np.dot(beta,x.T),dtype = np.float64))

def likelihood(X,y,beta):
	'''
	@param X: the sample variables matrix
	@param y: the sample label matrix
	@param beta: the parameter vector in 3.27
	@return: the log-likelihood of 3.27
	'''
	sum_0 = 0
	m,n = np.shape(X)
	for i in range(m):
		sum_0 += likelihood_sub(X[i],y[i],beta)

	return sum_0

def parital_derivative(X,y,beta):#refer to 3.30 
	'''
	@param X: the sample variables matrix
	@param y: the sample label matrix
	@param beta: the parameter vector in 3.27
	@return: the partial derivative of beta[j]
	'''
	m,n = np.shape(X)
	pd = np.zeros(n)

	for i in range(m):
		#the original code seems to be wrong because of the ommittance of '-'
		tmp = -(y[i]-sigmoid(X[i],beta))
		for j in range(n):
			pd[j] += X[i][j]*tmp
	return pd

'''foundational gradDscent algorithm'''
def gradDscent_1(X,y):
	'''
	@param X: X is the variable matrix
	@param y: y is the label array
	@return: the best parameter estimate of 3.27
	'''
	h = 0.1#step length of iterator
	max_times = 500#give the iterative times limit
	m,n = np.shape(X)

	#to show the curve depicting the parameters' covnergence progress 
	b = np.zeros((n,max_times))
	beta = np.zeros(n)#initialize parameter
	delta_beta = np.ones(n)*h
	llh = 0
	llh_temp = 0

	for i in range(max_times):
		beta_temp = beta

		for j in range(n):
			#for partial derivative
			beta[j] += delta_beta[j]
			llh_temp = likelihood(X,y,beta)
			theta = 1/(1+1/np.exp(i*n+j,dtype = np.float64))
			delta_beta[j] = -h*(llh_temp-llh)/delta_beta[j]+theta*delta_beta[j]

			b[j,i] = beta[j]

			beta[j] = beta_temp[j]

		beta += delta_beta
		llh = likelihood(X,y,beta)

	t = np.arange(max_times)

	f2 = plt.figure(3)

	p1 = plt.subplot(311)
	p1.plot(t,b[0])
	plt.ylabel('w1')

	p2 = plt.subplot(312)
	p2.plot(t,b[1])
	plt.ylabel('w2')

	p3 = plt.subplot(313)
	p3.plot(t,b[2])
	plt.ylabel('b')

	plt.show()
	return beta

#implementation of stochastic gradDscent algorithms
def gradDscent_2(X,y):
	'''
	@param X: X is the variable matrix
	@param y: y is the label array
	@return: the best parameter estimate of 3.27
	'''
	m,n = np.shape(X)
	h = 0.5#step length of iterator and initial
	beta = np.zeros(n)#parameter and initial
	delta_beta = np.ones(n)*h
	llh = 0
	llh_temp = 0;
	b = np.zeros((n,m))#for show convergence curve of parameter

	for i in range(m):
		beta_temp = beta

		for j in range(n):
			#for partial derivative
			theta = 1/(1+1/math.exp(i*n+j))
			beta[j] += delta_beta[j]

			b[j,i] = beta[j]
			llh_temp = likelihood_sub(X[i],y[i],beta)
			delta_beta[j] = -h*(llh_temp-llh)/delta_beta[j]+theta*delta_beta[j]

		beta += delta_beta
		llh = likelihood_sub(X[i],y[i],beta)#that's the key step:to take just one sample at a time

	t = np.arange(m)
	f2 = plt.figure(3)

	p1 = plt.subplot(311)
	p1.plot(t,b[0])
	plt.ylabel('w1')

	p2 = plt.subplot(312)
	p2.plot(t,b[1])
	plt.ylabel('w2')

	p3 = plt.subplot(313)
	p3.plot(t,b[2])
	plt.ylabel('b')

	plt.show()

	return beta

def sigmoid(x,beta):
	'''
	@param x: the variable to be predicted
	@param beta:the parameter
	@return: the sigmoid function value
	'''
	return 1.0/(1+np.exp(-np.dot(beta,x.T),dtype = np.float64))

def predict(X,beta):
	'''
	predict the label class using sigmoid func
	@param X:data sample in the form of [data_vector,label]
	@param beta: the param of sigmoid in the form of[w,b]
	@return: the class label array
	'''
	m,n = np.shape(X)
	y = np.zeros(m)

	for i in range(m):
		if sigmoid(X[i],beta) > 0.5:
			y[i] = 1
	return y




