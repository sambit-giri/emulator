import numpy as np
#from sklearn.model_selection import KFold
from scipy.integrate import simps
import warnings 
warnings.filterwarnings("ignore")
from . import distances
from . import helper_functions as hf
from . import bayesian_optimisation as bopt
from . import sampling_space as smp 
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import pickle

class GPRemul:
	def __init__(self, simulator, prior, bounds, gpr=None, verbose=True, N=100, sampling='LHS', param_file=None, output_file=None, full_save_file=None):
		'''
		sampling: 'LHS', 'LHS_nsphere', 'MCS', 'MCS_nsphere'
		'''
		#self.N_init  = N_init
		#self.N_max  = N_max
		self.N = N
		self.simulator = simulator
		#self.distance  = distance
		self.verbose = verbose
		self.param_names = [kk for kk in prior]
		self.param_bound = bounds
		self.bounds = np.array([bounds[kk] for kk in self.param_names])

		if sampling.lower()=='lhs': self.sampling = smp.LH_sampling
		elif sampling.lower()=='lhs_nsphere': self.sampling = smp.LHS_nsphere
		elif sampling.lower()=='mcs': self.sampling = smp.MC_sampling
		elif sampling.lower()=='mcs_nsphere': self.sampling = smp.MCS_nsphere
		else: self.sampling = sampling

		self.setup_gpr(gpr=gpr)
		self.param_file  = param_file
		self.output_file = output_file
		self.full_save_file = full_save_file

		self.train_out = None
		self.params = None

	def setup_gpr(self, gpr=None, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5):
		if gpr is not None:
			self.gpr = gpr
		else:
			if kernel is None: kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
			self.gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer)

	def create_params(self, samples):
		n_params = self.bounds.shape[0]
		#samples  = self.N
		mins, maxs = self.bounds.min(axis=1), self.bounds.max(axis=1)
		params = self.sampling(n_params=n_params, samples=samples, mins=mins, maxs=maxs, outfile=None)
		return params

	def get_params(self, param_file=None, save_file=None):
		if param_file is not None: self.param_file = param_file
		if self.param_file is not None: 
			if isinstance(self.param_file, (str)):
				self.params = np.loadtxt(self.param_file)
			elif isinstance(self.param_file, (np.ndarray)):
				self.params = self.param_file
			else:
				print('Please provide the params as a text file or numpy array.')
		else:
			if self.params is None: self.params = self.create_params(self.N)
		if save_file is not None:
			np.savetxt(save_file.split('.txt')[0]+'.txt', self.params)

	def get_training_set(self, train_file=None, save_file=None):
		if train_file is None: self.train_file = train_file
		if self.train_file is not None: 
			if isinstance(self.train_file, (str)):
				self.train_out = np.loadtxt(self.train_file)
			elif isinstance(self.train_file, (np.ndarray)):
				self.train_out = self.train_file
			else:
				print('Please provide the training set outputs as a text file or numpy array.')
		else:
			if self.train_out is None: self.train_out = np.array([self.simulator(i) for i in self.params])
		if save_file is not None:
			np.savetxt(save_file.split('.txt')[0]+'.txt', self.train_out)

	def get_testing_set(self, test_file=None, test_param_file=None, save_file=None, N=50):
		if test_param_file is None: self.test_param_file = test_param_file
		if self.test_param_file is not None: 
			if isinstance(self.test_param_file, (str)):
				self.test_params = np.loadtxt(self.test_param_file)
			elif isinstance(self.test_file, (np.ndarray)):
				self.test_params = self.test_param_file
			else:
				print('Please provide the testing set outputs as a text file or numpy array.')
		else:
			self.test_params = self.create_params(N)
		
		if test_file is None: self.test_file = test_file
		if self.test_file is not None: 
			if isinstance(self.test_file, (str)):
				self.test_out = np.loadtxt(self.test_file)
			elif isinstance(self.test_file, (np.ndarray)):
				self.test_out = self.test_file
			else:
				print('Please provide the testing set outputs as a text file or numpy array.')
		else:
			self.test_out = np.array([self.simulator(i) for i in self.test_params])
		if save_file is not None:
			np.savetxt(test_file.split('.txt')[0]+'.txt', self.test_out)
			np.savetxt(test_param_file.split('.txt')[0]+'.txt', self.test_params)

	def save_simulations(self, full_save_file=None):
		if self.full_save_file is None: self.full_save_file = full_save_file
		data = {'params': self.params, 'outputs': self.train_out}
		pickle.dump(data, open(self.full_save_file.split('.pkl')[0]+'.pkl', 'wb'))

	def save_model(self, model_file):
		pickle.dump(self.gpr, open(model_file.split('.pkl')[0]+'.pkl', 'wb'))

	def run(self):
		## Setup the training set.
		self.get_params()
		self.get_training_set()
		## Training with GPR
		self.gpr.fit(self.params, self.train_out)
		scr = self.gpr.score(self.params, self.train_out)
		print('Score: {0:.3f}'.format(scr))
		## Testing
		# Update this


class GPRemul_BO:
	def __init__(self, simulator, prior, bounds, gpr=None, verbose=True, N=100, N_init=10, sampling='LHS', param_file=None, output_file=None, full_save_file=None, exploitation_exploration=1000):
		'''
		sampling: 'LHS', 'LHS_nsphere', 'MCS', 'MCS_nsphere'
		'''
		self.N_init  = N_init
		self.N = N
		self.simulator = simulator
		#self.distance  = distance
		self.verbose = verbose
		self.param_names = [kk for kk in prior]
		self.param_bound = bounds
		self.bounds = np.array([bounds[kk] for kk in self.param_names])

		if sampling.lower()=='lhs': self.sampling = smp.LH_sampling
		elif sampling.lower()=='lhs_nsphere': self.sampling = smp.LHS_nsphere
		elif sampling.lower()=='mcs': self.sampling = smp.MC_sampling
		elif sampling.lower()=='mcs_nsphere': self.sampling = smp.MCS_nsphere
		else: self.sampling = sampling

		self.setup_gpr(gpr = gpr)
		self.param_file  = param_file
		self.output_file = output_file
		self.full_save_file = full_save_file

		self.train_out = None
		self.get_params()
		self.params = self.params[:self.N_init,:]
		self.exploitation_exploration = exploitation_exploration

	def setup_gpr(self, gpr=None, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5):
		if gpr is not None:
			self.gpr = gpr
		else:
			if kernel is None: kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
			self.gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer)

	def create_params(self, samples):
		n_params = self.bounds.shape[0]
		#samples  = self.N
		mins, maxs = self.bounds.min(axis=1), self.bounds.max(axis=1)
		params = self.sampling(n_params=n_params, samples=samples, mins=mins, maxs=maxs, outfile=None)
		return params

	def get_params(self, param_file=None, save_file=None):
		if param_file is not None: self.param_file = param_file
		if self.param_file is not None: 
			if isinstance(self.param_file, (str)):
				self.params = np.loadtxt(self.param_file)
			elif isinstance(self.param_file, (np.ndarray)):
				self.params = self.param_file
			else:
				print('Please provide the params as a text file or numpy array.')
		else:
			if self.params is None: self.params = self.create_params(self.N)
		if save_file is not None:
			np.savetxt(save_file.split('.txt')[0]+'.txt', self.params)

	def get_training_set_init(self, train_file=None, save_file=None):
		if train_file is None: self.train_file = train_file
		if self.train_file is not None: 
			if isinstance(self.train_file, (str)):
				self.train_out = np.loadtxt(self.train_file)
			elif isinstance(self.train_file, (np.ndarray)):
				self.train_out = self.train_file
			else:
				print('Please provide the training set outputs as a text file or numpy array.')
		else:
			if self.train_out is None: self.train_out = np.array([self.simulator(i) for i in self.params])
		if save_file is not None:
			np.savetxt(save_file.split('.txt')[0]+'.txt', self.train_out)

	def get_testing_set(self, test_file=None, test_param_file=None, save_file=None, N=50):
		if test_param_file is None: self.test_param_file = test_param_file
		if self.test_param_file is not None: 
			if isinstance(self.test_param_file, (str)):
				self.test_params = np.loadtxt(self.test_param_file)
			elif isinstance(self.test_file, (np.ndarray)):
				self.test_params = self.test_param_file
			else:
				print('Please provide the testing set outputs as a text file or numpy array.')
		else:
			self.test_params = self.create_params(N)
		
		if test_file is None: self.test_file = test_file
		if self.test_file is not None: 
			if isinstance(self.test_file, (str)):
				self.test_out = np.loadtxt(self.test_file)
			elif isinstance(self.test_file, (np.ndarray)):
				self.test_out = self.test_file
			else:
				print('Please provide the testing set outputs as a text file or numpy array.')
		else:
			self.test_out = np.array([self.simulator(i) for i in self.test_params])
		if save_file is not None:
			np.savetxt(test_file.split('.txt')[0]+'.txt', self.test_out)
			np.savetxt(test_param_file.split('.txt')[0]+'.txt', self.test_params)

	def save_simulations(self, full_save_file=None):
		if self.full_save_file is None: self.full_save_file = full_save_file
		data = {'params': self.params, 'outputs': self.train_out}
		pickle.dump(data, open(self.full_save_file.split('.pkl')[0]+'.pkl', 'wb'))

	def save_model(self, model_file):
		pickle.dump(self.gpr, open(model_file.split('.pkl')[0]+'.pkl', 'wb'))

	def run(self, N=None):
		if N is not None: self.N = N
		## Initialise
		if self.train_out is not None:
			if self.train_out.shape[0]<self.N_init:
				self.get_training_set_init()
		else:
			self.get_training_set_init()
		## Further sampling
		start_iter = len(self.params)
		for n_iter in range(start_iter,self.N):
			self.gpr.fit(self.params, self.train_out)
			X_next = bopt.propose_location(bopt.negativeGP_LCB_definedmu, self.params, self.train_out, self.gpr, self.bounds, n_restarts=10, xi=self.exploitation_exploration).T
			y_next = self.simulator(X_next[0])
			self.params = np.vstack((self.params,X_next))
			self.train_out = np.vstack((self.train_out,y_next))

		## Training with GPR
		self.gpr.fit(self.params, self.train_out)
		y_pred, y_std = self.gpr.predict(self.params, return_std=True)
		self.y_pred = y_pred
		self.y_std  = y_std
		scr = self.gpr.score(self.params, self.train_out)
		print('Score: {0:.3f}'.format(scr))
		## Testing
		# Update this

