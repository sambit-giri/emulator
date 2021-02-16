import numpy as np
import pickle
from . import helper_functions as hf
from time import time, sleep
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try: import tensorflow as tf
except: print('Install Tensorflow to use prob_nn.')

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

try: 
    import torch
    from torch import nn 
except: 
    print('Install PyTorch.')

# ##### For GPU #######
# if torch.cuda.is_available():
#     model.cuda()

class DenseNet(nn.Module):
    def __init__(self, d_layer, activation_func=None, epochs=1000, verbose=True, 
                learning_rate=None, loss_fn=None, batch_norm=False, dropout=0.5,
                optimizer='Adam'
                ):
        super().__init__()
        self.d_layer = d_layer
        self.epochs  = epochs
        self.verbose = verbose
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss(reduction='sum')
        self.learning_rate = learning_rate if learning_rate is not None else 1e-4
        self.optimizer = optimizer if optimizer is not None else 'Adam' 

        self.batch_norm = batch_norm
        if 0<=dropout<=1:
            self.dropout = dropout
        else:
            self.dropout = False
            print('For applying dropout, provide a number between 0 and 1.')

        if activation_func is None: activation_func = [nn.ReLU()]
        if len(activation_func)<=len(self.d_layer):
            self.activation_func = []
            for i in range(len(self.d_layer)):
                self.activation_func.append(activation_func[0])
        else:
            self.activation_func = activation_func

        self.clear_model()


    def prepare_model(self, input_dim, output_dim):
        modules = [nn.Linear(input_dim, self.d_layer[0])]
        if self.batch_norm: modules.append(nn.BatchNorm1d(self.d_layer[0]))
        if self.dropout: modules.append(nn.Dropout(self.dropout))
        modules.append(self.activation_func[0])

        for i in range(len(self.d_layer)-1):
            modules.append(nn.Linear(self.d_layer[i], self.d_layer[i+1]))
            if self.batch_norm: modules.append(nn.BatchNorm1d(self.d_layer[i+1]))
            if self.dropout: modules.append(nn.Dropout(self.dropout))
            modules.append(self.activation_func[i+1])

        modules.append(nn.Linear(self.d_layer[-1], output_dim))

        self.model = nn.Sequential(*modules)

    def clear_model(self):
        self.model  = None
        self.losses = []

    def fit(self, X_train, y_train, epochs=None):
        if epochs is not None: self.epochs = epochs
        if type(X_train)==np.ndarray:
            if X_train.ndim<=1: X_train = X_train[:,None]
            X_train = X_train.astype(np.float32)
            X_train = torch.from_numpy(X_train)
        if type(y_train)==np.ndarray:
            if y_train.ndim<=1: y_train = y_train[:,None]
            y_train = y_train.astype(np.float32)
            y_train = torch.from_numpy(y_train)

        input_dim, output_dim = X_train.shape[1], y_train.shape[1]
        if self.model is None: 
            self.prepare_model(input_dim, output_dim)
            epoch_start = 0
        else:
            epoch_start = len(self.losses)
            print('Found model which was run for {} epochs.'.format(epoch_start))
            print('To restart, run clear_model() before fitting.')
            sleep(0.5)

        if self.optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.optimizer=='RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        if self.optimizer=='Adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)

        pbar = tqdm(range(epoch_start,self.epochs)) if self.verbose else range(epoch_start,self.epochs)
        for t in pbar:
            # Forward pass: compute predicted y by passing x to the model. Module objects
            # override the __call__ operator so you can call them like functions. When
            # doing so you pass a Tensor of input data to the Module and it produces
            # a Tensor of output data.
            y_pred = self.model(X_train)

            # Compute and print loss. We pass Tensors containing the predicted and true
            # values of y, and the loss function returns a Tensor containing the
            # loss.
            loss = self.loss_fn(y_pred, y_train)
            self.losses.append(loss.item())
            # if t % 100 == 99:
            #     print(t+1, loss.item())
            if self.verbose:
                pbar.set_description("Epoch={0:} | loss={1:.3f}".format(t+1, loss.item()))

            # Zero the gradients before running the backward pass.
            self.model.zero_grad()

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            loss.backward()

            # Update the weights using gradient descent. Each parameter is a Tensor, so
            # we can access its gradients like we did before.
            # with torch.no_grad():
            #     for param in self.model.parameters():
            #         param -= self.learning_rate * param.grad
            self.optimizer.step()

    def predict(self, X_test):
        if type(X_test)==np.ndarray:
            if X_test.ndim<=1: X_test = X_test[:,None]
            X_test = X_test.astype(np.float32)
            X_test = torch.from_numpy(X_test)
        if self.model is None:
            print('Fit model before predicting.')
            return None
        y_pred = self.model(X_test)
        return y_pred.cpu().data.numpy()
    
    def score(self, X_test, y_test, metric='r2'):
        y_pred = self.predict(X_test)
        scr = metric_function(y_test, y_pred, metric=metric)
        return scr


def save_model(model, filename):
    pickle.dump(model, open(filename,'wb'))
    print('Model parameters are saved as', filename)

def load_model(filename):
    model = pickle.load(open(filename,'rb'))
    return model


def metric_function(y_true, y_pred, metric='r2'):
    from sklearn import metrics

    if type(metric)==str:
        assert metric in ['explained_variance_score', 'max_error', 
                          'mean_squared_error', 'mean_squared_log_error',
                          'mean_absolute_error', 'median_absolute_error',
                          'r2_score', 'r2']

        if metric=='explained_variance_score': metric = metrics.explained_variance_score
        if metric=='max_error': metric = metrics.max_error
        if metric=='mean_squared_error': metric = metrics.mean_squared_error
        if metric=='mean_squared_log_error': metric = metrics.mean_squared_log_error
        if metric=='median_absolute_error': metric = metrics.median_absolute_error
        if metric in ['r2', 'r2_score']: metric = metrics.r2_score

    return metric(y_true, y_pred)




