import numpy as np
import matplotlib.pyplot as mp
import time
from NeuralNetwork import Neural_Network



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


print('Training')
startTime = time.clock()
# Training data
scalar = 3.
nn = Neural_Network()
cost = nn.costFunction(tin,tout)
costArray = cost
while cost > 0.01:
	dJdW1,dJdW2 = nn.costFunctionPrime(tin,tout)
	nn.W1 = nn.W1 - scalar*(dJdW1)
	nn.W2 = nn.W2 - scalar*(dJdW2)
	cost = nn.costFunction(tin,tout)
	print(cost)
	costArray = np.append(costArray,cost)	

endTime = time.clock()	
print('Total Time :',endTime - startTime)
print(tin)
print(nn.yHat)
# Plot graph
np.savetxt('W1.out',nn.W1)
np.savetxt('W2.out',nn.W2)

fout = open('C:\\Users\\WahSeng\\Desktop\\Neural Network Tutorial\\W1.txt','w')
fout.write(str(nn.W1))
fout.close()

fout = open('C:\\Users\\WahSeng\\Desktop\\Neural Network Tutorial\\W2.txt','w')
fout.write(str(nn.W2))
fout.close()

mp.plot( - 1/np.log(costArray),'b-^')
mp.title('Cost function')
mp.xlabel('Iteration')
mp.ylabel('Cost')
mp.show()


print(nn.forward([800/maxTin1,1/maxTin2])*maxTout)






