import numpy as np
from sklearn.metrics import r2_score
import pickle
from . import helper_functions as hf
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

try: import GPy
except: print('Install GPy to use GPR_GPy and SparseGPR_GPy.')

try: import torch
except: print('Install PyTorch.')

try:
	import pyro
	import pyro.contrib.gp as gp
	import pyro.distributions as dist
except:
	print('Install Pyro to use GPR_pyro.')

try:
	import gpytorch
except:
	print('Install gpytorch to use GPR_GPyTorch.')


class GPR_GPy:
    def __init__(self, max_iter=1000, max_f_eval=1000, kernel=None, verbose=True, n_restarts_optimizer=5, n_jobs=0):
        # define kernel
        self.kernel     = kernel
        self.max_iter   = max_iter
        self.max_f_eval = max_f_eval
        self.verbose    = verbose
        self.n_jobs     = n_jobs
        self.n_restarts_optimizer = n_restarts_optimizer
    
    def fit(self, X_train, y_train):
        # check kernel
        if self.kernel is None:
            print('Setting kernel to Matern32.')
            input_dim = X_train.shape[1]
            # self.kernel = GPy.kern.Matern52(input_dim,ARD=True)
            self.kernel = GPy.kern.Matern32(input_dim,ARD=True)
        # create simple GP model
        self.m = GPy.models.GPRegression(X_train,y_train,self.kernel)
        # optimize
        if self.n_restarts_optimizer:
            self.m.optimize_restarts(
                num_restarts=self.n_restarts_optimizer,
                robust=False,
                #verbose=self.verbose,
                messages=self.verbose,
                parallel=True if self.n_jobs else False,
                num_processes=self.n_jobs if self.n_jobs else None,
                max_f_eval=self.max_f_eval,
                max_iters=self.max_iter,
                )
        else:
            self.m.optimize(messages=self.verbose, max_f_eval=self.max_f_eval)
        
    def predict(self, X_test, return_std=False):
        y_pred, y_var = self.m.predict(X_test)
        if return_std: return y_pred, np.sqrt(y_var)
        return y_pred
    
    def score(self, X_test, y_test):
        y_pred, y_std = self.m.predict(X_test)
        scr = r2_score(y_test, y_pred)
        return scr

    def save_model(self, filename, save_trainset=True):
        # np.save(filename, self.m.param_array)
        save_dict = {'kernel': self.m.kern.to_dict(), 'param_array': self.m.param_array}
        if save_trainset:
            save_dict['X'] = np.array(self.m.X)
            save_dict['Y'] = np.array(self.m.Y)
        pickle.dump(save_dict, open(filename, 'wb'))
        print('Model parameters are saved.')

    def load_model(self, filename, X=None, Y=None):
        load_dict = pickle.load(open(filename, 'rb'))
        self.kernel = GPy.kern.Kern.from_dict(load_dict['kernel'])
        # self.num_inducing = load_dict['num_inducing']
        if 'X' in load_dict.keys() and 'Y' in load_dict.keys():
            X = load_dict['X']
            Y = load_dict['Y']
        else:
            print('The file does not contain the training data.')
            print('Please provide it to the load_model through X and Y parameters.')
            return None
        
        m_load = GPy.models.GPRegression(X, Y, initialize=False, kernel=self.kernel)
        m_load.update_model(False)
        m_load.initialize_parameter()
        m_load[:] = load_dict['param_array']
        m_load.update_model(True)
        self.m = m_load
        return m_load


class SparseGPR_GPy:
    def __init__(self, max_iter=1000, max_f_eval=1000, kernel=None, verbose=True, n_restarts_optimizer=5, n_jobs=0, num_inducing=10):
        # define kernel
        self.kernel     = kernel
        self.max_iter   = max_iter
        self.max_f_eval = max_f_eval
        self.verbose    = verbose
        self.n_jobs     = n_jobs
        self.n_restarts_optimizer = n_restarts_optimizer
        self.num_inducing = num_inducing

    def setup_model(self, X_train, y_train):
        input_dim = X_train.shape[1]
        # check kernel
        if self.kernel is None:
            print('Setting kernel to Matern32.')
            # self.kernel = GPy.kern.Matern52(input_dim,ARD=True)
            self.kernel = GPy.kern.Matern32(input_dim,ARD=True)

        # define inducing points
        # self.Z = np.random.rand(self.num_inducing,input_dim)*(X_train.max(axis=0)-X_train.min(axis=0))+X_train.min(axis=0)

        # create simple GP model
        # self.m = GPy.models.SparseGPRegression(X_train,y_train,Z=self.Z,kernel=self.kernel)
        self.m = GPy.models.SparseGPRegression(X_train,y_train,num_inducing=self.num_inducing,kernel=self.kernel)

    def fit(self, X_train, y_train):
        self.setup_model(X_train, y_train)
        
        # optimize
        if self.n_restarts_optimizer:
            self.m.optimize_restarts(
                num_restarts=self.n_restarts_optimizer,
                robust=False,
                #verbose=self.verbose,
                messages=self.verbose,
                parallel=True if self.n_jobs else False,
                num_processes=self.n_jobs if self.n_jobs else None,
                max_f_eval=self.max_f_eval,
                max_iters=self.max_iter,
                )
        else:
            self.m.optimize(messages=self.verbose, max_f_eval=self.max_f_eval)
        # if self.verbose:
        #     print(self.m)
        return self.m
        
    def predict(self, X_test, return_std=False):
        y_pred, y_std = self.m.predict(X_test)
        if return_std: return y_pred, y_std
        return y_pred
    
    def score(self, X_test, y_test):
        y_pred, y_std = self.m.predict(X_test)
        scr = r2_score(y_test, y_pred)
        return scr

    def save_model(self, filename, save_trainset=True):
    	# np.save(filename, self.m.param_array)
    	save_dict = {'kernel': self.m.kern.to_dict(), 'param_array': self.m.param_array, 'num_inducing': self.num_inducing}
    	if save_trainset:
    		save_dict['X'] = np.array(self.m.X)
    		save_dict['Y'] = np.array(self.m.Y)
    	pickle.dump(save_dict, open(filename, 'wb'))
    	print('Model parameters are saved.')

    def load_model(self, filename, X=None, Y=None):
    	load_dict = pickle.load(open(filename, 'rb'))
    	self.kernel = GPy.kern.Kern.from_dict(load_dict['kernel'])
    	self.num_inducing = load_dict['num_inducing']
    	if 'X' in load_dict.keys() and 'Y' in load_dict.keys():
    		X = load_dict['X']
    		Y = load_dict['Y']
    	else:
    		print('The file does not contain the training data.')
    		print('Please provide it to the load_model through X and Y parameters.')
    		return None
    	
    	m_load = GPy.models.SparseGPRegression(X, Y, initialize=False, num_inducing=self.num_inducing, kernel=self.kernel)
    	m_load.update_model(False)
    	m_load.initialize_parameter()
    	m_load[:] = load_dict['param_array']
    	m_load.update_model(True)
    	self.m = m_load
    	return m_load
    	

class SVGPR_GPy:
    def __init__(self, max_iter=1000, max_f_eval=1000, kernel=None, verbose=True, n_restarts_optimizer=5, n_jobs=0, num_inducing=10):
        # define kernel
        self.kernel     = kernel
        self.max_iter   = max_iter
        self.max_f_eval = max_f_eval
        self.verbose    = verbose
        self.n_jobs     = n_jobs
        self.n_restarts_optimizer = n_restarts_optimizer
        self.num_inducing = num_inducing
    
    def fit(self, X_train, y_train):
        input_dim = X_train.shape[1]
        # check kernel
        if self.kernel is None:
            print('Setting kernel to Matern32.')
            # self.kernel = GPy.kern.Matern52(input_dim,ARD=True)
            self.kernel = GPy.kern.Matern32(input_dim,ARD=True)

        # define inducing points
        #self.Z = np.random.rand(self.num_inducing,input_dim)*(X_train.max(axis=0)-X_train.min(axis=0))+X_train.min(axis=0)

        # create simple GP model
        self.m = GPy.models.SparseGPRegression(X,y,num_inducing=self.num_inducing,kernel=self.kernel)

        # optimize
        if self.n_restarts_optimizer:
            self.m.optimize_restarts(
                num_restarts=self.n_restarts_optimizer,
                robust=False,
                #verbose=self.verbose,
                messages=self.verbose,
                parallel=True if self.n_jobs else False,
                num_processes=self.n_jobs if self.n_jobs else None,
                max_f_eval=self.max_f_eval,
                max_iters=self.max_iter,
                )
        else:
            self.m.optimize(messages=self.verbose, max_f_eval=self.max_f_eval)
        
    def predict(self, X_test, return_std=False):
        y_pred, y_std = self.m.predict(X_test)
        if return_std: return y_pred, y_std
        return y_pred
    
    def score(self, X_test, y_test):
        y_pred, y_std = self.m.predict(X_test)
        scr = r2_score(y_test, y_pred)
        return scr

class GPR_pyro:
	def __init__(self, max_iter=1000, tol=0.01, kernel=None, loss_fn=None, verbose=True, n_restarts_optimizer=5, n_jobs=0, estimate_method='MLE', learning_rate=1e-3):
		# define kernel
		self.kernel     = kernel
		self.max_iter   = max_iter
		self.verbose    = verbose
		self.n_jobs     = n_jobs
		self.n_restarts_optimizer = n_restarts_optimizer
		self.estimate_method = estimate_method
		self.learning_rate   = learning_rate
		self.loss_fn = loss_fn
		self.tol     = tol

	def fit(self, train_x, train_y):
		if type(train_x)==np.ndarray: train_x = torch.from_numpy(train_x)
		if type(train_y)==np.ndarray: train_y = torch.from_numpy(train_y)
		# check kernel
		if self.kernel is None:
			print('Setting kernel to Matern32.')
			input_dim = train_x.shape[1]
			self.kernel = gp.kernels.Matern32(input_dim, variance=None, lengthscale=None, active_dims=None)

		# create simple GP model
		self.model = gp.models.GPRegression(train_x, train_y, kernel)

		# optimize
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
		if self.loss_fn is None: self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
		self.losses = np.array([])
		n_wait, max_wait = 0, 5

		for i in range(self.max_iter):
			self.optimizer.zero_grad()
			loss = self.loss_fn(self.model.model, self.model.guide)
			loss.backward()
			self.optimizer.step()
			self.losses = np.append(self.losses,loss.item()) 
			print(i+1, loss.item())
			dloss = self.losses[-1]-self.losses[-2]    			
			if 0<=dloss and dloss<self.tol: n_wait += 1
			else: n_wait = 0
			if self.n_wait>=self.max_wait: break

	def predict(self, X_test, return_std=True, return_cov=False):
		y_mean, y_cov = self.model(X_test, full_cov=True, noiseless=False)

		if return_std: 
			y_std = cov.diag().sqrt()
			return y_pred, y_std
		if return_cov: return y_pred, y_cov
		return y_pred

	def score(self, X_test, y_test):
		y_pred = self.predict(X_test, return_std=False, return_cov=False)
		scr = r2_score(y_test, y_pred)
		return scr

class SparseGPR_pyro:
    def __init__(self, max_iter=1000, tol=0.001, kernel=None, error_fn=None, loss_fn=None, verbose=True, n_Xu=10, n_jobs=0, estimate_method='MLE', learning_rate=1e-3, method='VFE', n_restarts_optimizer=5, validation=0.1):
        # define kernel
        self.kernel     = kernel
        self.max_iter   = max_iter
        self.verbose    = verbose
        self.n_jobs     = n_jobs
        self.n_restarts_optimizer = n_restarts_optimizer
        self.estimate_method = estimate_method
        self.learning_rate   = learning_rate
        self.loss_fn = loss_fn
        self.tol     = tol
        self.n_Xu    = n_Xu
        self.method  = method
        self.error_fn   = mean_squared_error if error_fn is None else error_fn
        self.validation = validation

        # # Initialise output
        self.model = None
        self.losses = None
        self.optimizer = None
        self.continue_run = False
        self.train_err = None
        self.valid_err = None

    def fit_1out(self, train_x, train_y, n_Xu=None, past_info=None):
        if n_Xu is not None: self.n_Xu = n_Xu

        if self.validation is not None:
            if type(train_x)!=np.ndarray: train_x = train_x.detach().numpy()
            if type(train_y)!=np.ndarray: train_y = train_y.detach().numpy()
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.validation, random_state=42)
            valid_x = torch.from_numpy(valid_x)

        if type(train_x)==np.ndarray: train_x = torch.from_numpy(train_x)
        if type(train_y)==np.ndarray: train_y = torch.from_numpy(train_y)
        # check kernel
        if self.kernel is None:
            print('Setting kernel to Matern32.')
            input_dim = train_x.shape[1]
            self.kernel = gp.kernels.Matern32(input_dim, variance=None, lengthscale=None, active_dims=None)

        self.Xu = np.linspace(train_x.min(axis=0)[0].data.numpy(), train_x.max(axis=0)[0].data.numpy(), self.n_Xu)
        self.Xu = torch.from_numpy(self.Xu)

        # create simple GP model
        model = gp.models.SparseGPRegression(train_x, train_y, self.kernel, Xu=self.Xu, jitter=1.0e-5, approx=self.method) if past_info is None else past_info['model']

        # optimize
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate) if past_info is None else past_info['optimizer']
        if self.loss_fn is None: self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        losses = np.array([]) if past_info is None else past_info['losses']

        tr_err, vl_err = 10000, 10000
        if self.validation is not None:
            train_err = np.array([]) if past_info is None else past_info['train_err']
            valid_err = np.array([]) if past_info is None else past_info['valid_err']

        n_wait, max_wait = 0, 5

        for i in range(losses.size,self.max_iter):
            optimizer.zero_grad()
            loss = self.loss_fn(model.model, model.guide)
            loss.backward()
            optimizer.step()
            losses = np.append(losses,loss.item()) 
            if self.validation is not None:
                # print(type(train_y))
                tr_err = self.error_fn(train_y.detach().numpy(), model(train_x, full_cov=False)[0].detach().numpy())
                vl_err = self.error_fn(valid_y, model(valid_x, full_cov=False)[0].detach().numpy())
                train_err = np.append(train_err, tr_err)
                valid_err = np.append(valid_err, vl_err)
            if self.verbose: 
                hf.loading_verbose('                                                                                            ')
                hf.loading_verbose('{0} | loss={1:.2f} | train_error={2:.3f} | validation_error={2:.3f}'.format(i+1, loss.item(), tr_err, vl_err))
            dloss = losses[-1]-losses[-2] if len(losses)>2 else self.tol*1000			
            if 0<=dloss and dloss<self.tol: n_wait += 1
            else: n_wait = 0
            if n_wait>=max_wait: break

        if self.validation is not None: return model, optimizer, losses, train_err, valid_err
        return model, optimizer, losses

    def fit(self, train_x, train_y, n_Xu=None):
        if n_Xu is not None: self.n_Xu = n_Xu

        if type(train_x)==np.ndarray: train_x = torch.from_numpy(train_x)
        if type(train_y)==np.ndarray: train_y = torch.from_numpy(train_y)
        # check kernel
        if self.kernel is None:
            print('Setting kernel to Matern32.')
            input_dim = train_x.shape[1]
            self.kernel = gp.kernels.Matern32(input_dim, variance=None, lengthscale=None, active_dims=None)

        if self.model is not None: self.continue_run = True

        tstart = time()
        if train_y.ndim==1:
            if self.validation is not None:
                past_info = {'model':self.model, 'losses':self.losses, 'optimizer':self.optimizer, 'train_err': self.train_err, 'valid_err':self.valid_err} if self.continue_run else None
                model, optimizer, losses, train_err, valid_err = self.fit_1out(train_x, train_y, past_info=past_info)
                self.model, self.optimizer, self.losses, self.train_err, self.valid_err = model, optimizer, losses, train_err, valid_err
                tend = time()
            else:
                past_info = {'model':self.model, 'losses':self.losses, 'optimizer':self.optimizer} if self.continue_run else None
                model, optimizer, losses = self.fit_1out(train_x, train_y, past_info=past_info)
                self.model, self.optimizer, self.losses = model, optimizer, losses
                tend = time()
            print('\n...done | Time elapsed: {:.2f} s'.format(tend-tstart))
        else:
            if self.validation is not None:
                if self.model is None:
                    self.model, self.optimizer, self.losses, self.train_err, self.valid_err = {}, {}, {}, {}, {}
                for i in range(train_y.shape[1]):
                    print('Regressing output variable {}'.format(i+1))
                    past_info = {'model':self.model[i], 'losses':self.losses[i], 'optimizer':self.optimizer[i], 'train_err': self.train_err[i], 'valid_err':self.valid_err[i]} if self.continue_run else None
                    model, optimizer, losses, train_err, valid_err = self.fit_1out(train_x, train_y[:,i], past_info=past_info)
                    self.model[i], self.optimizer[i], self.losses[i], self.train_err[i], self.valid_err[i] = model, optimizer, losses, train_err, valid_err
                    tend = time()
                    print('\n...done | Time elapsed: {:.2f} s'.format(tend-tstart))
            else:
                if self.model is None:
                    self.model, self.optimizer, self.losses = {}, {}, {}
                for i in range(train_y.shape[1]):
                    print('Regressing output variable {}'.format(i+1))
                    past_info = {'model':self.model[i], 'losses':self.losses[i], 'optimizer':self.optimizer[i]} if self.continue_run else None
                    model, optimizer, losses = self.fit_1out(train_x, train_y[:,i], past_info=past_info)
                    self.model[i], self.optimizer[i], self.losses[i] = model, optimizer, losses
                    tend = time()
                    print('\n...done | Time elapsed: {:.2f} s'.format(tend-tstart))


    def predict_1out(self, X_test, return_std=True, return_cov=False):
        if type(X_test)==np.ndarray: X_test = torch.from_numpy(X_test)
        
        y_mean, y_cov = self.model(X_test, full_cov=True, noiseless=False)

        if return_std: 
            y_std = y_cov.diag().sqrt()
            return y_mean.detach().numpy(), y_std.detach().numpy()
        if return_cov: return y_mean.detach().numpy(), y_cov.detach().numpy()
        return y_mean.detach().numpy()

    def predict(self, X_test, return_std=True, return_cov=False):
        if type(X_test)==np.ndarray: X_test = torch.from_numpy(X_test)
        if type(self.model) is dict:
            y_mean, y_cov = [], []
            for i in range(len(self.model)):
                y_mean0, y_cov0 = self.model[i](X_test, full_cov=True, noiseless=False)
                y_mean.append(y_mean0.detach().numpy())
                y_cov.append(y_cov0.detach().numpy())
            if return_std:
                y_std = [np.sqrt(np.diag(y_cov1)) for y_cov1 in y_cov]
                return np.array(y_mean).T, np.array(y_std).T
            if return_cov: return np.array(y_mean).T, np.array(y_cov).T
            return np.array(y_mean).T


    def score(self, X_test, y_test):
        if type(X_test)==np.ndarray: X_test = torch.from_numpy(X_test)
        if type(y_test)==torch.Tensor: y_test = y_test.detach().numpy()

        y_pred = self.predict(X_test, return_std=False, return_cov=False)
        scr = r2_score(y_test, y_pred)
        return scr


class GPR_GPyTorch:
    def __init__(self, max_iter=1000, tol=0.01, kernel=None, loss_fn=None, verbose=True, learning_rate=1e-3, optimizer=None, validation=0.1):
        # define kernel
        self.kernel     = kernel
        self.max_iter   = max_iter
        self.verbose    = verbose
        self.learning_rate   = learning_rate
        self.loss_fn = loss_fn
        self.tol     = tol
        self.optimizer  = optimizer
        # self.validation = validation

        self.train_loss = []
        self.valid_loss = []

    def prepare_model(self, train_x, train_y, kernel=None):
        multi_task = False
        if train_y.ndim>1:
            if train_y.shape[1]>1:
                multi_task = True

        if multi_task:
            print('Model for Multivariate output.')
            # We will use the GP model for multivariate output, exact inference
            class MultitaskGPModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.MultitaskMean(
                        gpytorch.means.ConstantMean(), num_tasks=train_y.shape[1]
                    )
                    self.covar_module = gpytorch.kernels.MultitaskKernel(
                        kernel, num_tasks=train_y.shape[1], rank=1
                    )

                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

            # initialize likelihood and model
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1])
            self.model = MultitaskGPModel(train_x, train_y, self.likelihood)
        else:
            # We will use the simplest form of GP model, exact inference
            class ExactGPModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            # initialize likelihood and model
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = ExactGPModel(train_x, train_y, self.likelihood)

    def fit(self, train_x, train_y):
        # if self.validation is not None:
        #     if type(train_x)!=np.ndarray: train_x = train_x.detach().numpy()
        #     if type(train_y)!=np.ndarray: train_y = train_y.detach().numpy()
        #     train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.validation, random_state=42)
        #     valid_x = torch.from_numpy(valid_x)

        if type(train_x)==np.ndarray: train_x = torch.from_numpy(train_x.astype(np.float32))
        if type(train_y)==np.ndarray: train_y = torch.from_numpy(train_y.astype(np.float32))
        print(train_x.shape, train_y.shape)

        # Check kernel
        if self.kernel is None:
            print('Setting kernel to Matern32.')
            self.kernel = gpytorch.kernels.MaternKernel(nu=1.5)

        # create simple GP model
        if len(self.train_loss)==0:
            self.prepare_model(train_x, train_y, kernel=self.kernel)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()


        if self.optimizer is None: 
            print('Using the adam optimizer.')
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        if self.loss_fn in [None, 'marginal_log_likelihood', 'mll']:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        else: 
            mll = self.loss_fn

        # optimize
        for i in range(len(self.train_loss),self.max_iter):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            #print(type(output))
            #print(output)
            # Calc loss and backprop gradients
            #print(output, train_y.shape)
            loss = -mll(output, train_y)
            loss.backward()
            self.train_loss.append(loss.item())

            # if self.validation:
            #     self.model.eval()
            #     self.likelihood.eval()

            #     valid_out = self.likelihood(self.model(valid_x))
            #     valid_ls  = -mll(valid_out, valid_y)
            #     self.valid_loss.append(valid_ls.item())
            #     print('Iter %d/%d - Train Loss: %.3f   Valid Loss: %.3f   ' % (
            #         i + 1, self.max_iter, self.train_loss[-1], self.valid_loss[-1]
            #     ))
            #     self.model.train()
            #     self.likelihood.train()
            # else:
            print('Iter %d/%d - Loss: %.3f   ' % (
                i + 1, self.max_iter, self.train_loss[-1]
            ))
            self.optimizer.step()

    def predict(self, X_test, return_ci=True):
        if type(X_test)==np.ndarray: X_test = torch.from_numpy(X_test.astype(np.float32))

        # Get into evaluation (predictive posterior) mode
        model, likelihood = self.model, self.likelihood
        model.eval()
        likelihood.eval()


        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(X_test))

        if return_ci:
            lower, upper = observed_pred.confidence_region()
            return observed_pred.mean.numpy(), lower.detach().numpy(), upper.detach().numpy()

        return observed_pred.detach().numpy()


    def score(self, X_test, y_test):
        if type(X_test)==np.ndarray: X_test = torch.from_numpy(X_test)
        if type(y_test)==torch.Tensor: y_test = y_test.detach().numpy()

        y_pred = self.predict(X_test, return_ci=False)
        scr = r2_score(y_test, y_pred)
        return scr


class SVGP_GPyTorch:
    def __init__(self, max_iter=1000, tol=0.01, kernel=None, loss_fn=None, verbose=True, learning_rate=1e-3, optimizer=None, validation=0.1):
        # define kernel
        self.kernel     = kernel
        self.max_iter   = max_iter
        self.verbose    = verbose
        self.learning_rate   = learning_rate
        self.loss_fn = loss_fn
        self.tol     = tol
        self.optimizer  = optimizer
        # self.validation = validation

        self.train_loss = []
        self.valid_loss = []

    def prepare_model(self, train_x, train_y, kernel=None):
        multi_task = False
        if train_y.ndim>1:
            if train_y.shape[1]>1:
                multi_task = True

        if multi_task:
            print('Model for Multivariate output.')
            # We will use the GP model for multivariate output, exact inference
            class MultitaskGPModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.MultitaskMean(
                        gpytorch.means.ConstantMean(), num_tasks=train_y.shape[1]
                    )
                    self.covar_module = gpytorch.kernels.MultitaskKernel(
                        kernel, num_tasks=train_y.shape[1], rank=1
                    )

                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

            # initialize likelihood and model
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1])
            self.model = MultitaskGPModel(train_x, train_y, self.likelihood)
        else:
            # We will use the simplest form of GP model, exact inference
            class ExactGPModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            # initialize likelihood and model
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = ExactGPModel(train_x, train_y, self.likelihood)

    def fit(self, train_x, train_y):
        # if self.validation is not None:
        #     if type(train_x)!=np.ndarray: train_x = train_x.detach().numpy()
        #     if type(train_y)!=np.ndarray: train_y = train_y.detach().numpy()
        #     train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.validation, random_state=42)
        #     valid_x = torch.from_numpy(valid_x)

        if type(train_x)==np.ndarray: train_x = torch.from_numpy(train_x.astype(np.float32))
        if type(train_y)==np.ndarray: train_y = torch.from_numpy(train_y.astype(np.float32))
        print(train_x.shape, train_y.shape)

        # Check kernel
        if self.kernel is None:
            print('Setting kernel to Matern32.')
            self.kernel = gpytorch.kernels.MaternKernel(nu=1.5)

        # create simple GP model
        if len(self.train_loss)==0:
            self.prepare_model(train_x, train_y, kernel=self.kernel)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()


        if self.optimizer is None: 
            print('Using the adam optimizer.')
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        if self.loss_fn in [None, 'marginal_log_likelihood', 'mll']:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        else: 
            mll = self.loss_fn

        # optimize
        for i in range(len(self.train_loss),self.max_iter):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            #print(type(output))
            #print(output)
            # Calc loss and backprop gradients
            #print(output, train_y.shape)
            loss = -mll(output, train_y)
            loss.backward()
            self.train_loss.append(loss.item())

            # if self.validation:
            #     self.model.eval()
            #     self.likelihood.eval()

            #     valid_out = self.likelihood(self.model(valid_x))
            #     valid_ls  = -mll(valid_out, valid_y)
            #     self.valid_loss.append(valid_ls.item())
            #     print('Iter %d/%d - Train Loss: %.3f   Valid Loss: %.3f   ' % (
            #         i + 1, self.max_iter, self.train_loss[-1], self.valid_loss[-1]
            #     ))
            #     self.model.train()
            #     self.likelihood.train()
            # else:
            print('Iter %d/%d - Loss: %.3f   ' % (
                i + 1, self.max_iter, self.train_loss[-1]
            ))
            self.optimizer.step()

    def predict(self, X_test, return_ci=True):
        if type(X_test)==np.ndarray: X_test = torch.from_numpy(X_test.astype(np.float32))

        # Get into evaluation (predictive posterior) mode
        model, likelihood = self.model, self.likelihood
        model.eval()
        likelihood.eval()


        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(X_test))

        if return_ci:
            lower, upper = observed_pred.confidence_region()
            return observed_pred.detach().numpy(), lower.detach().numpy(), upper.detach().numpy()

        return observed_pred.detach().numpy()


    def score(self, X_test, y_test):
        if type(X_test)==np.ndarray: X_test = torch.from_numpy(X_test)
        if type(y_test)==torch.Tensor: y_test = y_test.detach().numpy()

        y_pred = self.predict(X_test, return_ci=False)
        scr = r2_score(y_test, y_pred)
        return scr
