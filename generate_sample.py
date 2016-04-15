import numpy as np
import sys
#sample number of annual visits based on age group
def getNumberVisits(ageGroup):
	mean = [3.7,3.2,3,2.9,3,3.6] #mean of poisson distribution for different age groups
	ind = -1
	#identifying age group
	if ageGroup <= 0.02: #12-17
		ind = 0
	elif ageGroup <= 0.27 and ageGroup > 0.02: #18-24
		ind = 1
	elif ageGroup <= 0.49 and ageGroup > 0.27 : #25-34
		ind = 2
	elif ageGroup <= 0.82 and ageGroup > 0.49 : #35-54
		ind = 3
	elif ageGroup <= 0.92 and ageGroup > 0.82: #55-64
		ind = 4
	else: #>65
		ind = 5
	sum = 0
	#simulating visits for a year 
	for i in range(12):
		sum += np.random.poisson(mean[ind])
	return [ind < 4,sum]	 #return indicator if younger than 55 and total number of visits for a year	
	
def getStops(n_shops,num_visits,sm55):
	#65% visits - specific, 35% - leisure	
	sp = 0
	leisure = 0	
	gender = np.random.rand(1) >= 0.5
	#modeling number of stops in a visit as a gamma process
	if gender:
		if sm55:
			sp_stops = np.random.gamma(2.057,0.957)
			leisure_stops = np.random.gamma(11.973,0.279)
		else:
			sp_stops = np.random.gamma(2.773,0.721)
			leisure_stops = np.random.gamma(64.515,0.035)
	else:
		if sm55:
			sp_stops = np.random.gamma(4.331,0.623)
			leisure_stops = np.random.gamma(711.11,0.006)
		else:
			sp_stops = np.random.gamma(1.897,1.270)
			leisure_stops = np.random.gamma(31.21,0.138)
	shop_visits = np.zeros(n_shops,dtype='float32')
	sp_stops = int(round(sp_stops)) # no. of specific stops
	leisure_stops = int(round(leisure_stops)) # no. of leisure stops
	sp_visits = int(round(0.65 * num_visits*sp_stops)) #total number  of specific stops
	leisure_visits = int(round(0.35 * num_visits*leisure_stops)) #total number of leisure stops
	sp_shops = np.random.random_integers(0,n_shops-1,min(sp_stops,n_shops)) # id of specific stops
	leisure_shops = np.random.random_integers(0,n_shops-1,min(leisure_stops,n_shops)) # id of leisure stops
	
	sp_distr = np.random.rand(sp_stops)
	leisure_distr = np.random.rand(leisure_stops)
	sp_distr = np.round(sp_visits*sp_distr/np.sum(sp_distr)) #distribution of specific stops among  specific shops
	leisure_distr = np.round(leisure_visits*leisure_distr/np.sum(leisure_distr)) # distribution of leisure stops among leisure shops
	ind = 0	
	for i in sp_shops:
		shop_visits[i] = min(num_visits,sp_distr[ind])
		ind += 1
	ind = 0	
	for i in leisure_shops:	
		shop_visits[i] = min(num_visits,leisure_distr[ind])
		ind += 1
	return shop_visits*10.0/num_visits	

if __name__ == "__main__":
#get number of visitors and number of shops from 
	n_visitors = int(sys.argv[1])	
	n_shops = int(sys.argv[2])
	data = np.zeros((n_visitors,n_shops+1),dtype='float32')
	values = np.random.rand(n_visitors)
	for i in range(n_visitors):
		tup = getNumberVisits(values[i])
		data[i,0] = tup[1]
		data[i,1:] = getStops(n_shops,tup[1],tup[0]) 
	data.astype('int')	 
	np.savetxt('simulated_data.csv',data,'%.2f',delimiter=',')
