import numpy as np
import matplotlib.pyplot as mp
import time
from NeuralNetwork import Neural_Network

W1 = np.loadtxt('W1.out')
W2 = np.loadtxt('W2.out')
W3 = np.loadtxt('W3.out')

nn= Neural_Network()

nn.SetW1(W1)
nn.SetW2(W2)
nn.SetW3(W3)

yHat = 0
for i in range(1,100):
	temp = nn.forward((i/100,1))
	yHat = np.append(yHat,temp)
	
	
# LOAD DATA	
fileStr = 'C:\\Users\\WahSeng\\Desktop\\Neural Network Tutorial\\TrainingData.txt'

# Open the file and read the contents
data = np.genfromtxt(fileStr)

# Load data
tin = data[:,0:2]
tout = data[:,2:3]

# Normalize data
maxTin1 = np.max(tin[:,0])
maxTin2 = np.max(tin[:,1])
maxTout = np.max(tout)
tin[:,0] = tin[:,0] / maxTin1
tin[:,1] = tin[:,1] / maxTin2
tout = tout / maxTout

#PLOT
n = np.linspace(0,1,100)

fun, ax = mp.subplots(2,sharex = True)
ax[0].plot(n,yHat,'b-',tin[:,0],tout,'r^')
ax[1].plot(n,yHat,'b-',tin[:,1],tout,'g^')
mp.show()