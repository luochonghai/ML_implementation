# -*- coding=utf-8 -*-
import math
from math import *
import matplotlib.pyplot as plt
import numpy as np

sigma = 4.0/sqrt(2.0*math.pi)

def phi(x):
	return (1.0+erf(x/sqrt(2.0)))/2.0

def norm_d(x):
	return exp(-x*x*math.pi/16.0)/4.0

def sigmoid(x):
	return 1.0/(1.0+exp(-x))

def f_sig(x):
	return log(math.e,1.0+exp(x))
x = np.arange(-3.0*sigma,3.0*sigma,0.05)
#sigmoid函数和正态分布函数N(0,sigma)的积分函数之差，用蓝色曲线表示
y = [sigmoid(xi)-phi(xi/sigma) for xi in x]
#sigmoid函数的导函数和正态分布函数之差，用红色曲线表示
y1 = [sigmoid(xi)*(1-sigmoid(xi))-norm_d(xi) for xi in x]

plt.plot(x,y,linewidth = 2)
plt.plot(x,y1,color = '#FF0000')
plt.show()
'''可以说，sigmoid函数就是正态分布函数的高仿版'''