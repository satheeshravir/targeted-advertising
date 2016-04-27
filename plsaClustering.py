# Code for computing clusters from user ratings
#  input: ratings file name in csv

import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.matlib
from sklearn.cross_validation import train_test_split
from scipy.stats import pearsonr

class plsaCluster:
	def __init__(self,k,ratings):
		self.k = k # number of clusters
		self.ratings = ratings # the ratings matrix
		dims = ratings.shape		
		self.numUsers = dims[0] # number of users
		self.numShops = dims[1] # number of shops
		self.nll = []  # negative log likelihood in each step
		#self.threshold = mu #threshold for probability of user belonging to a cluster
	
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
		joint_ui = -np.log(np.dot(self.probUZ,np.dot(np.diag(self.probZ),self.probSZ.T)))
		self.nll.append(np.sum(joint_ui*self.ratings))
		print 'Starting log likelihood ' + str(self.nll[-1])
		for i in range(30):
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
		#self.sanityCheck()		
	
	# check all attributes satisfy sanity check	
	def sanityCheck(self):
		print 'number of Clusters: ' + str(self.k)
		print 'number of Users: ' + str(self.numUsers)
		print 'number of Shops: ' + str(self.numShops)
		print 'sum of Cluster probabilities: ' + str(np.sum(self.probZ))
		print 'sum of probabilities of user given cluster: ' + str(np.sum(self.probUZ))
		print 'sum of prior Cluster probabilities: ' + str(np.sum(self.probSZ))
	
	# compute rating vector for each cluster
	def getClusterRatings(self,threshold):
		cratings = np.zeros((self.k,self.numShops),dtype='float32')
		probUZ = np.copy(self.probUZ)		
		probUZ[probUZ < threshold] = 0 #set membership of all users below threshold to zero
		for i in range(self.k):
			if(np.sum(probUZ[:,i])==0):
				return None
		
		for i in range(self.k):
			cratings[i,:] = np.dot(self.ratings.T,probUZ[:,i])/np.sum(probUZ[:,i]) #compute aggregated rating
		return cratings

	def saveModel(self,cvnumber):		
		#np.savetxt('GUI_code/model',self.cratings,'%.2f')	
		np.savetxt('plsafiles',self.probUZ,'%.7f')

#return index of shop ratings to hide for each user to compute MAE
def hideShopRatings(test):
	dims = test.shape
	nrows = dims[0]
	shop_index = []
	for i in range(nrows):
		shop_count = np.sum(test[i,:]>0)
		shop_id = np.random.randint(0,shop_count)
		for j in range(dims[1]):
			if(test[i,j] > 0):
				if(shop_id == 0):
					shop_index.append(j)
					break
				else:
					shop_id -=1
	return shop_index

def computeMAE(cratings,shop_index,X_test):
	
	#compute mae
	nrows = len(shop_index)
	dims = cratings.shape
	nClusters = dims[0]
	nShops = dims[1]
	mae = 0
	for i in range(nrows):
		original_rating = float(X_test[i,shop_index[i]])
		user_rating = np.copy(X_test[i,:])
		user_rating[shop_index[i]] = 0
		if(np.var(user_rating)==0):
			user_rating[0] += 0.01
		pRating = [0]*nClusters
		for k in range(nClusters):
			out = pearsonr(cratings[k,:],user_rating)
			pRating[k] = out[0]
		cOrder = [j[0] for j in sorted(enumerate(pRating),key=lambda x:x[1],reverse=True)] #order clusters based on pearson coefficient
		nRating = (cratings[cOrder[0],:] + cratings[cOrder[1],:] + cratings[cOrder[2],:])/3 # neighbourhood rating
		predicted_rating = nRating[shop_index[i]]
		mae += abs(original_rating-predicted_rating)
	return mae/nrows

def tuneThreshold(k,all_ratings):
	thresholds = [0.0008,0.0007,0.0006,0.0005,0.0004,0.0003,0.0002,0.0001]
	nt = len(thresholds)
	mae = np.zeros(8,dtype='float32')
	dim = all_ratings.shape
	nUsers = dim[0]
	#use cross validation to select best threshold
	for i in range(5):
		print "Cross Validation iteration " + str(i+1) 
		#split training and test data
		X_train,X_test = train_test_split(all_ratings,test_size=0.2,random_state=42)
		#create test data
		shop_index = hideShopRatings(X_test)
		#compute the cluster values
		cluster = plsaCluster(k,X_train)
		cluster.initializeParameters()
		cluster.learnParameters()
		#compute mae for different thresholds
		for t in range(nt):
			cratings = cluster.getClusterRatings(thresholds[t])
			if(cratings == None):
				mae[t] += 100
			else:
				mae[t] += computeMAE(cratings,shop_index,X_test) 
	print mae
	mae = mae/5
	plt.plot(thresholds,mae)
	plt.xlabel('Threshold')
	plt.ylabel('Mean Average Error')
	plt.show()

#tune model for different thresholds
def tuneK(threshold,all_ratings):
		
if __name__ == "__main__":
	data = np.loadtxt(sys.argv[1],delimiter=',')
	#cratings = np.loadtxt('GUI_code/model',delimiter=' ')
	#sidex = hideShopRatings(data[:,1:])	
	#print computeMAE(cratings,sidex,data[:,1:])
	tuneThreshold(15,data[:,1:])	
	#cluster1 = plsaCluster(15,data[:,1:])
	#cluster1.initializeParameters()
	#cluster1.learnParameters()	
	#plt.plot(range(len(cluster1.nll)),cluster1.nll)
	#plt.xlabel('Iteration')
	#plt.ylabel('Negative log likelihood')	
	#plt.show()
	#cluster1.computeClusterRatings()
	#cluster1.saveModel()
	
