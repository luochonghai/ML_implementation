import numpy as np

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
	return np.tanh(x)

def tanh_prime(x):
	return 1.0-x**2

class NeuralNetwork:
	def __init__(self,layers,activation = 'tanh'):
		if activation == 'sigmoid':
			self.activation = sigmoid
			self.activation_prime = sigmoid_prime
		elif activation == 'tanh':
			self.activation = tanh
			self.activation_prime = tanh_prime

		'''set weights'''
		self.weights = []

		'''layers = [2,2,1]
		range of weight values(-1,1)
		input and hidden layers:random(3X3)
		'''
		for i in range(1,len(layers)-1):
			r = 2*np.random.random((layers[i-1]+1,layers[i]+1))-1
			s = np.random.random((layers[i-1]+1,layers[i]+1))-1
			self.weights.append(r)

		'''output layer:random(3X1)'''
		r = 2*np.random.random((layers[i]+1,layers[i+1]))-1
		self.weights.append(r)

	def fit(self,X,y,learning_rate = 0.2,epochs = 100000):
		'''add column of ones to X
		to add bias unit to input layer
		'''
		'''View inputs as arrays with at least two dimensions.'''
		ones = np.atleast_2d(np.ones(X.shape[0]))
		X = np.concatenate((ones.T,X),axis = 1)

		for k in range(epochs):
			if k%10000 == 0:
				print('epochs:',k)

			i = np.random.randint(X.shape[0])
			a = [X[i]]

			for l in range(len(self.weights)):
				dot_value = np.dot(a[l],self.weights[l])
				activation = self.activation(dot_value)
				a.append(activation)

			'''output layer'''
			error = y[i]-a[-1]
			deltas = [error*self.activation_prime(a[-1])]

			'''to begin at the second to last layer
			(a layer before the output layer)'''
			for l in range(len(a)-2,0,-1):
				deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

			'''reverse
			[level3(output)->levle2[hidden]] => [level2(hidden)->level3(output)]
			'''
			deltas.reverse()

			'''back propagation
			1.multiply its output delta and input activation to get 
			the gradient of the weight
			2.subtract a ratio(percentage) of the gradient from the weight
			'''
			for i in range(len(self.weights)):
				layer = np.atleast_2d(a[i])
				delta = np.atleast_2d(deltas[i])
				self.weights[i] += learning_rate*layer.T.dot(delta)

	def predict(self,x):
		a = np.concatenate((np.ones(1).T,np.array(x)),axis = 0)
		for l in range(0,len(self.weights)):
			a = self.activation(np.dot(a,self.weights[l]))
		return a		

if __name__ == '__main__':
	nn = NeuralNetwork([2,2,1])

	X = np.array([[0,0],
				[0,1],
				[1,0],
				[1,1]])

	y = np.array([0,1,1,0])

	nn.fit(X,y)

	for e in X:
		print(e,nn.predict(e))


