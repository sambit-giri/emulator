import numpy as np
from sklearn.metrics import r2_score
import pickle
from . import helper_functions as hf
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

try: import tensorflow as tf
except: print('Install Tensorflow.')

try:
	import probflow as pf
except:
	print('Install probflow to use prob_nn.')


class prob_DenseNN:
    def __init__(self, d_layer, epochs=1000, verbose=True, heteroscedastic=False, n_jobs=1, lr=None, flipout=True, optimizer=None, optimizer_kwargs={}):
        # define kernel
        self.d_layer = d_layer
        self.epochs  = epochs
        self.verbose = verbose
        self.heteroscedastic = heteroscedastic
        self.n_jobs = n_jobs if n_jobs>1 else None 
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr      = lr
        self.flipout = flipout

    def prepare_model(self, input_dim, output_dim):
        dlr = [input_dim] + self.d_layer + [output_dim]
        self.model = pf.DenseRegression(dlr, heteroscedastic=self.heteroscedastic)
    
    def fit(self, X_train, y_train, epochs=None):
        if epochs is not None: self.epochs = epochs
        if X_train.ndim<=1: X_train = X_train[:,None]
        if y_train.ndim<=1: y_train = y_train[:,None]
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        input_dim, output_dim = X_train.shape[1], y_train.shape[1]
        self.prepare_model(input_dim, output_dim)

        callbacks = []
        if self.verbose:
        	callbacks.append(pf.callbacks.MonitorMetric('mse', X_train, y_train))

        self.model.fit(X_train, y_train, 
                num_workers=self.n_jobs, 
                optimizer=self.optimizer, 
                optimizer_kwargs=self.optimizer_kwargs,
                lr=self.lr,
                epochs=self.epochs,
                flipout=self.flipout,
                callbacks=callbacks,
                )

    def predict(self, X_test, ci=0.95):
        if X_test.ndim>1: X_test = X_test[:,None]
        X_test = X_test.astype(np.float32)
        y_pred = self.model.predict(X_test)
        return y_pred

    def predictive_interval(self, X_test, ci=0.95):
        lb, ub = self.model.predictive_interval(X_test, ci=ci)
        return lb, ub
    
    def score(self, X_test, y_test, metric='r2'):
        scr = self.model.metric(metric, X_test, y_test)
        return scr

    def save_model(self, filename):
        if filename.split('.')[-1] != 'pfm': filename = filename+'.pfm'
        self.model.save(filename)
        print('Model parameters are saved as', filename)

    def load_model(self, filename):
        self.model = pf.load(filename)
        return self.model



