#!/usr/bin/python
import numpy as np
import pandas as pd
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer

data = ""
def loadData(file_path):
	global data
	input_file = open(file_path,"r")	
	data = np.loadtxt(file_path, delimiter = ',')
	results = np.copy(data)
	data = pd.DataFrame(data)
	df = pd.DataFrame(results)
	return df



def trainModel(data_frame, k):
	train = data_frame[data_frame[k] != 0.00]
	Y = train[k]
	X = train.drop(k,1)
	test = data_frame[data_frame[k] == 0.00].drop(k,1)
	return X.as_matrix(), Y.as_matrix(), test.as_matrix(), test.index.values

def nnTrain(data_frame):
	for k in range(1,21):
		X, Y, test, indices = trainModel(df,k)
		ds = SupervisedDataSet( 20, 1 )
		i=0
		for i in range(len(X)):
			ds.addSample(X[i],Y[i])

		if len(X) > 0:
			net = buildNetwork(20, 40, 1, bias=True, hiddenclass=TanhLayer)
			trainer = BackpropTrainer(net, ds)
			trainer.train()
			#trainer.trainEpochs(epochs=10)
			#trainer.trainUntilConvergence(maxEpochs=200)
			for i in range(len(test)):
			    data.set_value(indices[i],k, net.activate(test[i]))

df = loadData("simulated_data.csv")
nnTrain(df)
print data
data.to_csv("dense_matrix.csv", header=False, index=False)


