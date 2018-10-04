import numpy as np
import pickle

class KCons(object):
	
	def __init__(self, K, X, Y, dist_func, estimators_list):
        
		self.K = K
		self.X = X
		self.Y = Y
		self.mu = None
		self.clusters = None
		self.mask = np.zeros(self.X.shape[0])
        
		self.dist = dist_func
		self.estimators_list = estimators_list
		
		np.random.seed(seed=5)

	def _dist_from_centers(self):
		cent = self.mu
		X = self.X
		self.DISTX = np.full(X.shape[0], np.inf)
		
		for i in range(X.shape[0]):
			self.DISTX[i] = min(self.DISTX[i], self.dist(self.X[ i, :-2], cent[-1][:-2]))
		D2 = self.DISTX
		#D2 = np.array([min([np.linalg.norm(x-c)**2 for c in cent]) for x in X])
		self.D2 = D2
 
	def _choose_next_center(self):
		
		self.probs = self.D2/self.D2.sum()
		self.cumprobs = self.probs.cumsum()
		r = np.random.random()
		ind = np.where(self.cumprobs >= r)[0][0]
		return(self.X[ind])
 
	def init_centers(self):
		i = np.random.randint(low=0, high=self.X.shape[0])
		self.mu = [self.X[i, :]]
		while len(self.mu) < self.K:
			self._dist_from_centers()	
			self.mu.append(self._choose_next_center())
			
	def predict_cluster(self):
		
		self.mu_classified = []
		mu = self.mu
		iter = 0
	
		for center in mu:
			score = 0
			for h in self.estimators_list:
				learner = pickle.loads(h[1])
				score += learner.predict(center.reshape(1, -1))
				iter += 1
						
			score = score / iter
			
			if score < 0.5 : 
				self.mu_classified.append(0)
			else:
				self.mu_classified.append(1)
						
	def _cluster_points(self):
		mu = self.mu
		clusters  = np.zeros(self.X.shape[0])
		for j in range(self.X.shape[0]):
			x = self.X[j, :-2]
            #bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
              #               for i in enumerate(mu)], key=lambda t:t[1])[0]
			bestmukey = min([(i[0], self.dist(x, mu[i[0]][:-2])) \
                             for i in enumerate(mu)], key=lambda t:t[1])[0] 
			clusters[j] = bestmukey

		self.clusters = clusters
		
	def _reevaluate_centers(self):
		clusters = self.clusters[ self.mask == 0]
		newmu = []
		
		X = self.X[ self.mask == 0]
		Y = self.Y[ self.mask == 0]
		
        
		for i in range(len(self.mu)):
			grp = self.mu_classified[i]
			x = X[(Y != grp) & (clusters == i)]
			if x.shape[0] > 0:
				newmu.append(np.mean(x, axis = 0))
			else:
				self.mask[ (self.clusters == i)] = 1
		self.mu = newmu

	def _has_converged(self):
		K = len(self.oldmu)
		return(set([tuple(a) for a in self.mu]) == \
		set([tuple(a) for a in self.oldmu])\
		and len(set([tuple(a) for a in self.mu])) == K)
 
	def find_centers(self):
        
		X = self.X
		K = self.K
		# Initialize to K random centers
		self.init_centers()
		self.oldmu = X[np.random.randint(low=0, high=X.shape[0], size=K)]
		
		iter = 0
        
		while (not self._has_converged()) & (iter < 20):
			
			self.oldmu = self.mu
			self.predict_cluster()
	
			self._cluster_points()
			
			iter += 1
			self._reevaluate_centers()

