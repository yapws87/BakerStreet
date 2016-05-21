import numpy as np
import matplotlib.pyplot as mp
import time
from NeuralNetwork import Neural_Network



fileStr = 'C:\\Users\\WahSeng\\Desktop\\Neural Network Tutorial\\TrainingData.txt'

# Open the file and read the contents
data = np.genfromtxt(fileStr)

# Load data
tin = np.array(data[:,0:2])
tout = np.array(data[:,2:3])

# Normalize data
maxTin1 = np.max(tin[:,0])
maxTin2 = np.max(tin[:,1])
maxTout = np.max(tout)
tin[:,0] = tin[:,0] / maxTin1
tin[:,1] = tin[:,1] / maxTin2
tout = tout / maxTout


#Load Data
W1 = np.loadtxt('W1.out')
W2 = np.loadtxt('W2.out')
W3 = np.loadtxt('W3.out')


print('Training')
startTime = time.clock()
# Training data

scalar =3.
nn = Neural_Network()
#nn.SetW1(W1)
#nn.SetW2(W2)
#nn.SetW3(W3)

n = np.linspace(0,1,100)
n2 = np.ones(100)
fun, ax = mp.subplots(2,sharex = True)

cost = nn.costFunction(tin,tout)
costArray = []
costArray = np.append(costArray,cost)
costDiff = 100
costDiff2 = 100
count = 0
while costDiff > 1e-15:
	
	if costDiff2 < 1e-8:
		scalar = scalar #+ 0.1
	elif costDiff2 > 1e-5:
		scalar = scalar #- 0.1
		
		
	dJdW1,dJdW2, dJdW3 = nn.costFunctionPrime(tin,tout)
	nn.W1 = nn.W1 - scalar*(dJdW1)
	nn.W2 = nn.W2 - scalar*(dJdW2)
	nn.W3 = nn.W3 - scalar*(dJdW3)
	cost = nn.costFunction(tin,tout)
	
	Diff = abs(cost - costArray[-1])
	costArray = np.append(costArray,cost)
	costDiff2 = abs(costDiff - Diff)
	costDiff = Diff
	
	
	print(cost,costDiff,costDiff2,scalar)

	#PLOT
	if count > 2000:
		ax[0].clear()
		ax[1].clear()
		yHat = nn.forward(np.array([n,n]).T)
		ax[0].plot(n,yHat,'b-',tin[:,0],tout,'r^')
		ax[1].plot(n,yHat,'b-',tin[:,1],tout,'g^')
		mp.pause(0.01)
		count = 0
	
	count = count + 1

endTime = time.clock()	
print('Total Time :',endTime - startTime)
print(tin)
print(nn.yHat)

# Plot graph
np.savetxt('W1.out',nn.W1)
np.savetxt('W2.out',nn.W2)
np.savetxt('W3.out',nn.W3)

fout = open('C:\\Users\\WahSeng\\Desktop\\Neural Network Tutorial\\W1.txt','w')
fout.write(str(nn.W1))
fout.close()

fout = open('C:\\Users\\WahSeng\\Desktop\\Neural Network Tutorial\\W2.txt','w')
fout.write(str(nn.W2))
fout.close()

#mp.plot( - 1/np.log(costArray),'b-^')
#mp.title('Cost function')
#mp.xlabel('Iteration')
#mp.ylabel('Cost')
#mp.show()

print(nn.forward([800/maxTin1,1/maxTin2])*maxTout)






