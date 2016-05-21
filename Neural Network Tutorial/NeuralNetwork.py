import numpy as np

class Neural_Network(object):
	def __init__(self):
		#Define HyperParameters
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3
		self.lambd = 0#1e-21
		
		self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize,self.hiddenLayerSize)
		self.W3 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
	
	def SetLayerSize(self,inputLayerSize,outputLayerSize,hiddenLayerSize):
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.hiddenLayerSize = hiddenLayerSize
	
	def SetW1(self,W1):
		if(W1.shape[0] != self.inputLayerSize ):
			print('row size must be',self.inputLayerSize)
			return 
		if(W1.shape[1] != self.hiddenLayerSize ):
			print('col size must be',self.hiddenLayerSize)
			return 
		self.W1 = np.array(W1)
		
	
	def SetW2(self,W2):
		if(W2.shape[0] != self.hiddenLayerSize ):
			print('row size must be',self.hiddenLayerSize)
			return 
		self.W2 = np.array(W2)
	
		
	def SetW3(self,W3):
		if(W3.shape[0] != self.hiddenLayerSize ):
			print('row size must be',self.hiddenLayerSize)
			return 
		self.W3 = np.array(W3,ndmin = 2).T
		
	def sigmoid(self,z):
		#Apply sigmoid activation function
		return  1 /( 1 + np.exp(-z))
	
	def sigmoidPrime(self,z):
		#Derivative of sigmoid function
		return np.exp(-z)/ ((1 + np.exp(-z))**2)
		
	def forward(self,X):
		#Propagate inputs through network

		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		
		self.z3 = np.dot(self.a2,self.W2)
		self.a3 = self.sigmoid(self.z3)
			
		self.z4 = np.dot(self.a3,self.W3)
		
		yHat = self.sigmoid(self.z4)
		return yHat
		
	
		
	def costFunctionPrime(self,X,y):
		#Compute derivative with respect to W1 and W2
		self.yHat = self.forward(X)
			
		#delta3 = np.multiply(-(y - self.yHat),self.sigmoidPrime(self.z3))
		#dJdW2 = np.dot(self.a2.T,delta3)/X.shape[0] + self.lambd*self.W2
		
		#delta2 = np.multiply(np.dot(delta3,self.W2.T),self.sigmoidPrime(self.z2))
		#dJdW1 = np.dot(X.T,delta2)/X.shape[0] + self.lambd*self.W1
			
		delta4 = np.multiply(-(y - self.yHat),self.sigmoidPrime(self.z4))	
		dJdW3 = np.dot(self.a3.T,delta4)/X.shape[0]# + self.lambd*self.W3
				
		delta3 = np.multiply(np.dot(delta4,self.W3.T),self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T,delta3)/X.shape[0]# + self.lambd*self.W2
		
		delta2 = np.multiply(np.dot(delta3,self.W2.T),self.sigmoidPrime(self.z2))
		dJdW1 = np.dot(X.T,delta2)/X.shape[0]# + self.lambd*self.W1

		return dJdW1, dJdW2, dJdW3
		
	def costFunction(self,X,y):
		#Compute cost function
		self.yHat = self.forward(X)
		
		cost = 0.5*((y - self.yHat)**2)/X.shape[0]
		#cost = cost + self.lambd/2 * (sum(self.W1**2)+sum(self.W2**2))
		return np.sum(cost)
		
