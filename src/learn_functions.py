import numpy as np 
from time import time, sleep
from tqdm import tqdm 
import pickle

from skopt import sampler as smp
from skopt import Space 

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LassoCV
from sklearn import metrics #.r2_score


class Fit_Function:
	'''
	Fitting a surrogate model using random forest regression.
	'''
	def __init__(self, func, n_samples, estimator, n_params=None, range_param=None, r2_tol=0.98, 
		sampling_method='LH', validation=0.1):
		'''
		Parameters:
			func (function): a function or simulator that is to be modelled. 
							 This function will be used to generate the training set.
			n_samples (int): Number of training points to generate.
			estimator: The estimator which is used to train.
					   See https://scikit-learn.org/stable/supervised_learning.html for estimators.
			n_params = None (int) : Number of parameters used by the function or simulator.
			range_param = None(dict): A dictionary containing the range of each parameter.
			r2_tol = 1e-2 (float): Maximum r2 score (coefficient of determination). 
			sampling_method = 'LH' (str or function): The method used for sampling the 
							parameter space.
			validation = 0.1 (float): Percentage of the total number of samples in 
							training set that will be used to validate the model. 
		Returns:
			Nothing
		'''
		assert n_params is not None or range_param is not None

		self.func = func
		self.n_samples = n_samples
		self.model = estimator

		if range_param is not None:
			self.range_param = range_param
			self.n_params    = len(range_param.keys())
		else:
			self.n_params    = n_params
			self.range_param = {i:[0,1] for i in range(n_params)} 
		self.r2_tol     = r2_tol
		self.validation = validation 

		self.sampling_method = sampling_method
		self.init_points = None
		self.X_train     = None
		self.y_train     = None

	def initial_sampling(self, sampling_method=None):
		if sampling_method is not None:
			self.sampling_method = sampling_method
		
		if self.sampling_method.lower() in ['lh', 'lhs', 'latin hypercube', 'lh-maximin']:
			if self.sampling_method.lower() in ['lh', 'lhs', 'latin hypercube']:
				print('Using maximin criteria for latin hypercube sampling.')
				print('To use classic, centered, correlation or ratio-optimised criteria, input sampling_method') 
				print('as lh-classic, lh-centered, lh-correlation and lh-ratio-optimised respectively.')
			samp = smp.Lhs(criterion="maximin", iterations=10000) 
		elif self.sampling_method.lower() in ['lh-classic']:
			samp = smp.Lhs(lhs_type="classic", criterion=None)
		elif self.sampling_method.lower() in ['lh-centered']:
			samp = smp.Lhs(lhs_type="centered", criterion=None)
		elif self.sampling_method.lower() in ['lh-correlation']:
			samp = smp.Lhs(criterion="correlation", iterations=10000)	
		elif self.sampling_method.lower() in ['lh-ratio-optimised']:
			samp = smp.Lhs(criterion="ratio", iterations=10000)
		elif self.sampling_method.lower() in ['sobol']:
			samp = smp.Sobol()
		elif self.sampling_method.lower() in ['halton']:
			samp = smp.Halton()
		elif self.sampling_method.lower() in ['hammersly']:
			samp = smp.Hammersly()
		elif self.sampling_method.lower() in ['grid']:
			samp = smp.Grid(border="include", use_full_layout=False)

		# print('Creating initial sampling...')
		# space = Space([(self.range_param[ke][0],self.range_param[ke][1]) for ke in self.range_param.keys()]) 
		space = Space([(0,self.n_samples) for ke in self.range_param.keys()]) 
		self.init_points = np.array(samp.generate(space.dimensions, self.n_samples)).astype(float)
		for i,ke in enumerate(self.range_param.keys()):
			self.init_points[:,i] = self.init_points[:,i]*(self.range_param[ke][1]-self.range_param[ke][0])/self.n_samples+self.range_param[ke][0]
		# print('...done')

	def create_training_set(self):
		if self.init_points is None:
			self.initial_sampling()

		self.X_train = self.init_points
		print('Creating training set...')
		sleep(0.5)
		#self.y_train = np.array([self.func(i) for i in tqdm(self.X_train)])
		self.y_train = self.func(self.X_train)
		if self.y_train.ndim==2:
			if self.y_train.shape[1]==1:
				self.y_train = self.y_train.flatten()
		sleep(0.5)
		print('...done')

	def create_testing_set(self):
		init_points1, n_samples1 = self.init_points, self.n_samples
		self.n_samples = int(self.validation*self.n_samples)
		if self.n_samples<10: self.n_samples = 10 
		self.initial_sampling()

		self.X_test = self.init_points
		# print('Creating testing set...')
		sleep(0.5)
		# self.y_test = np.array([self.func(i) for i in tqdm(self.X_test)])
		self.y_test = self.func(self.X_test)
		if self.y_test.ndim==2:
			if self.y_test.shape[1]==1:
				self.y_test = self.y_test.flatten()
		sleep(0.5)
		# print('...done')
		self.init_points, self.n_samples = init_points1, n_samples1

	def model_func(self):
		if self.y_train is None or self.X_train is None:
			self.create_training_set()

		# self.model = RandomForestRegressor(n_estimators=int(self.n_samples/2))
		self.model.fit(self.X_train, self.y_train)
		self.modelled_func = lambda x: self.model.predict(x[:,None] if x.ndim==1 else x)

		if self.validation is not None:
			n_restarts = 5
			r2_scores  = []
			print('Testing model...')
			for i in range(n_restarts):
				self.create_testing_set()
				y_pred = self.modelled_func(self.X_test)
				r2_scores.append(metrics.r2_score(self.y_test, y_pred))
			self.r2_score = np.array(r2_scores).min()
			print('...done')
			print('r2 score:', self.r2_score)
			if self.r2_score<self.r2_tol:
				print('The training may not be large enough to give good accuracy.')
				print('Either increase the n_samples value or use a different parameter sampling recipe.')

	def save_model(self, filename):
		pickle.dump(self.model, open(filename, 'wb'))
		print('Model saved.')

	def load_model(self, filename):
		self.model = pickle.load(open(filename, 'rb'))
		self.modelled_func = lambda x: self.model.predict(x[:,None] if x.ndim==1 else x)
		print('Model loaded.')


class Fit_LassoCV(Fit_Function):
	'''
	Fitting a surrogate model using LASSO regression.
	'''
	def __init__(self, func, n_samples, n_params=None, range_param=None, r2_tol=0.98, 
		sampling_method='LH', validation=0.1, cv=5):
		'''
		Parameters:
			func (function): a function or simulator that is to be modelled. 
							 This function will be used to generate the training set.
			n_samples (int): Number of training points to generate.
			n_params = None (int) : Number of parameters used by the function or simulator.
			range_param = None(dict): A dictionary containing the range of each parameter.
			r2_tol = 1e-2 (float): Maximum r2 score (coefficient of determination). 
			sampling_method = 'LH' (str or function): The method used for sampling the 
							parameter space.
			validation = 0.1 (float): Percentage of the total number of samples in 
							training set that will be used to validate the model. 
			cv = 5 (int): Value for k-fold cross validation.
		Returns:
			Nothing
		'''
		assert n_params is not None or range_param is not None

		self.func = func
		self.n_samples   = n_samples
		if range_param is not None:
			self.range_param = range_param
			self.n_params    = len(range_param.keys())
		else:
			self.n_params    = n_params
			self.range_param = {i:[0,1] for i in range(n_params)} 
		self.r2_tol     = r2_tol
		self.validation = validation 

		self.sampling_method = sampling_method
		self.init_points = None
		self.X_train     = None
		self.y_train     = None

		self.model = LassoCV(cv=cv)
		super().__init__(func, n_samples, self.model, n_params=self.n_params, range_param=self.range_param, r2_tol=self.r2_tol, 
					sampling_method=self.sampling_method, validation=self.validation)

class Fit_GPR(Fit_Function):
	'''
	Fitting a surrogate model using Gaussian Process regression.
	'''
	def __init__(self, func, n_samples, n_params=None, range_param=None, r2_tol=0.98, 
		sampling_method='LH', validation=0.1, kernel=None):
		'''
		Parameters:
			func (function): a function or simulator that is to be modelled. 
							 This function will be used to generate the training set.
			n_samples (int): Number of training points to generate.
			n_params = None (int) : Number of parameters used by the function or simulator.
			range_param = None(dict): A dictionary containing the range of each parameter.
			r2_tol = 1e-2 (float): Maximum r2 score (coefficient of determination). 
			sampling_method = 'LH' (str or function): The method used for sampling the 
							parameter space.
			validation = 0.1 (float): Percentage of the total number of samples in 
							training set that will be used to validate the model. 
			kernel = None (int): Value for k-fold cross validation.
		Returns:
			Nothing
		'''
		assert n_params is not None or range_param is not None

		self.func = func
		self.n_samples   = n_samples
		if range_param is not None:
			self.range_param = range_param
			self.n_params    = len(range_param.keys())
		else:
			self.n_params    = n_params
			self.range_param = {i:[0,1] for i in range(n_params)} 
		self.r2_tol     = r2_tol
		self.validation = validation 

		self.sampling_method = sampling_method
		self.init_points = None
		self.X_train     = None
		self.y_train     = None

		self.model = GaussianProcessRegressor(kernel=kernel)
		super().__init__(func, n_samples, self.model, n_params=self.n_params, range_param=self.range_param, r2_tol=self.r2_tol, 
					sampling_method=self.sampling_method, validation=self.validation)

class Fit_RandomForest(Fit_Function):
	'''
	Fitting a surrogate model using random forest regression.
	'''
	def __init__(self, func, n_samples, n_params=None, range_param=None, r2_tol=0.98, 
		sampling_method='LH', validation=0.1):
		'''
		Parameters:
			func (function): a function or simulator that is to be modelled. 
							 This function will be used to generate the training set.
			n_samples (int): Number of training points to generate.
			n_params = None (int) : Number of parameters used by the function or simulator.
			range_param = None(dict): A dictionary containing the range of each parameter.
			r2_tol = 1e-2 (float): Maximum r2 score (coefficient of determination). 
			sampling_method = 'LH' (str or function): The method used for sampling the 
							parameter space.
			validation = 0.1 (float): Percentage of the total number of samples in 
							training set that will be used to validate the model. 
		Returns:
			Nothing
		'''
		assert n_params is not None or range_param is not None

		self.func = func
		self.n_samples   = n_samples
		if range_param is not None:
			self.range_param = range_param
			self.n_params    = len(range_param.keys())
		else:
			self.n_params    = n_params
			self.range_param = {i:[0,1] for i in range(n_params)} 
		self.r2_tol     = r2_tol
		self.validation = validation 

		self.sampling_method = sampling_method
		self.init_points = None
		self.X_train     = None
		self.y_train     = None

		self.model = RandomForestRegressor(n_estimators=int(self.n_samples/2))
		super().__init__(func, n_samples, self.model, n_params=self.n_params, range_param=self.range_param, r2_tol=self.r2_tol, 
					sampling_method=self.sampling_method, validation=self.validation)


