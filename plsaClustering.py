# Code for computing clusters from user ratings
# input k : number of clusters , ratings file name in csv

import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.matlib
class plsaCluster:
	def __init__(self,k,ratings):
		self.k = k # number of clusters
		self.ratings = ratings # the ratings matrix
		dims = ratings.shape		
		self.numUsers = dims[0] # number of users
		self.numShops = dims[1] # number of shops
		self.nll = []  # negative log likelihood in each step
	
	# random initialization of parameters to estimate	
	def initializeParameters(self):
		probZ = np.random.rand(self.k) # marginal proability of each cluster
		self.probZ = probZ / np.sum(probZ)
		probUZ = np.random.rand(self.numUsers,self.k) # probability of a user given cluster
		probSZ = np.random.rand(self.numShops,self.k) # probability of a shop given cluster
		norm_u = numpy.matlib.repmat(np.sum(probUZ,axis = 0),self.numUsers,1)
		norm_s = numpy.matlib.repmat(np.sum(probSZ,axis = 0),self.numShops,1)		
		self.probUZ = probUZ / norm_u
		self.probSZ = probSZ / norm_s
			 
	def learnParameters(self):
		probZUS = np.zeros((self.numUsers,self.numShops,self.k),dtype='float32') 
		nll_diff = 100
		joint_ui = -np.log(np.dot(self.probUZ,np.dot(np.diag(self.probZ),self.probSZ.T)))
		self.nll.append(np.sum(joint_ui*self.ratings))
		print 'Starting log likelihood ' + str(self.nll[-1])
		while nll_diff > 0.001:
			# E-step: Compute posterior probabilities
			for u in range(self.numUsers):
				for s in range(self.numShops):
					probZUS[u,s,:] = self.probZ * self.probUZ[u,:] * self.probSZ[s,:]
					probZUS[u,s,:] = probZUS[u,s,:] / np.sum(probZUS[u,s,:])
			
			# M-step: Update model parameters 
			rsum = np.sum(np.sum(self.ratings))
			for i in range(self.k):
				product = self.ratings * probZUS[:,:,i]					
				product_sum = np.sum(product)					
				self.probZ[i] = product_sum
				self.probUZ[:,i] = np.sum(product,axis=1) / product_sum
				self.probSZ[:,i] = np.sum(product,axis=0)/product_sum
			self.probZ[:] /= rsum
			
			#compute negative log likelihood
			joint_ui = -np.log(np.dot(self.probUZ,np.dot(np.diag(self.probZ),self.probSZ.T)))
			self.nll.append(np.sum(joint_ui*self.ratings))
			print 'iteration ' + str(len(self.nll)-1) + ' log likelihood ' + str(self.nll[-1])
			nll_diff = abs(self.nll[-2] - self.nll[-1])/self.nll[-2]
		self.sanityCheck()		
	#check all attributes satisfy sanity check	
	def sanityCheck(self):
		print 'number of Clusters: ' + str(self.k)
		print 'number of Users: ' + str(self.numUsers)
		print 'number of Shops: ' + str(self.numShops)
		print 'sum of Cluster probabilities: ' + str(np.sum(self.probZ))
		print 'sum of probabilities of user given cluster: ' + str(np.sum(self.probUZ))
		print 'sum of prior Cluster probabilities: ' + str(np.sum(self.probSZ))
		

if __name__ == "__main__":
	data = np.loadtxt(sys.argv[2],delimiter=',')
	cluster1 = plsaCluster(int(sys.argv[1]),data[:,1:])
	cluster1.initializeParameters()
	cluster1.learnParameters()	
	plt.plot(range(len(cluster1.nll)),cluster1.nll)
	plt.xlabel('Iteration')
	plt.ylabel('Negative log likelihood')	
	plt.show()
	
	
