# -*- coding: utf-8 -*

import numpy as np
'''get the projective point(2D) of a point to a line
	@param point: the coordinate of the point form as [a,b]
	@param line: the line parameters in the form of [k,t] which
	means y = k*x+t
	@return: the coordinate of the projective point
	'''

def GetProjectivePoint(point,line):
	a = point[0]
	b = point[1]
	k = line[0]
	t = line[1]

	if k == 0:
		return [a,t]
	elif k == np.inf:
		return [0,b]
	x = (a+k*(b-t))/(k*k+1)
	y = k*x+t
	return [x,y]