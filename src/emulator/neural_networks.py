import numpy as np
import pickle, copy
from time import time, sleep
from tqdm import tqdm

from . import helper_functions as hf
from .helper_functions import moving_average

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try: import tensorflow as tf
except: print('Install Tensorflow to use prob_nn.')

try: import probflow as pf
except: print('Install probflow to use prob_nn.')

try: 
    import torch
    from torch import nn 
    import torch.optim as optim
except: 
    print('Install PyTorch.')


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

class NNRegressor:
    def __init__(self, model=None, layers=[3,32,64,128,250],
                 dropout_prob=0.5, weight_decay=0,
                 optimizer='Adam', loss_fn='MSE',
                 learning_rate=1e-4,
                 ):
        if model is None:
            layer_modules = []
            for i in range(len(layers) - 1):
                layer_modules.append(nn.Linear(layers[i], layers[i+1]))
                if i < len(layers) - 2:
                    layer_modules.append(nn.ReLU())
            model = nn.Sequential(*layer_modules)

        self.model = model 
        self.X_min = None
        self.X_max = None
        self.y_min = None
        self.y_max = None
        self.n_epochs = None
        self.batch_size = None
        self.lr = None
        self.loss_history = None
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

    def numpy_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32)
    
    def tensor_to_numpy(self, data):
        return data.numpy()
    
    def normalise(self, data):
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_normed = (data-data_min)/(data_max-data_min)
        return {'min': data_min, 'max': data_max, 'normed': data_normed}
    
    def apply_dropout(self):
        if self.dropout_prob > 0:
            for layer in self.model.modules():
                if isinstance(layer, nn.Dropout):
                    layer.p = self.dropout_prob

    def apply_weight_decay(self):
        if self.weight_decay > 0:
            for param in self.model.parameters():
                param.data -= self.weight_decay * param.data

    def fit(self, X, y, n_epochs=100, batch_size=10, learning_rate=None):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        if learning_rate is not None:
            self.learning_rate = learning_rate 

        self.X_min, self.X_max = X.min(axis=0), X.max(axis=0)
        self.y_min, self.y_max = y.min(axis=0), y.max(axis=0)
        X_normed = (X-self.X_min)/(self.X_max-self.X_min)
        y_normed = (y-self.y_min)/(self.y_max-self.y_min)


        X_train, X_test, y_train, y_test = train_test_split(
                X_normed, y_normed, train_size=0.8, shuffle=True)
        X_train = self.numpy_to_tensor(X_train)
        y_train = self.numpy_to_tensor(y_train)
        X_test = self.numpy_to_tensor(X_test)
        y_test = self.numpy_to_tensor(y_test)

        model = self.model
        batch_start = torch.arange(0, len(X_train), batch_size)

        # loss function and optimizer
        if self.loss_fn.lower()=='mse':
            loss_fn = nn.MSELoss()  # mean square error
        else:
            loss_fn = self.loss_fn

        if self.optimizer=='Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer=='RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer=='Adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = self.optimizer

        # Hold the best model
        best_mse = np.inf   # init to infinity
        best_weights = None
        history = []
        
        # training loop
        for epoch in range(n_epochs):
            model.train()
            with tqdm(batch_start, unit="batch", mininterval=0) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train[start:start+batch_size]
                    y_batch = y_train[start:start+batch_size]
                    # forward pass
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(mse=float(loss))
            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(X_test)
            mse = loss_fn(y_pred, y_test)
            mse = float(mse)
            history.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(model.state_dict())
        
        # restore model and return best accuracy
        model.load_state_dict(best_weights)
        self.model = model
        self.loss_history = history

    def predict(self, X):
        X_normed = (X-self.X_min)/(self.X_max-self.X_min)
        # Convert input data to tensor
        X_tensor = self.numpy_to_tensor(X_normed)
        # Set model to evaluation mode
        self.model.eval()
        # Make predictions
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        # Convert predictions to numpy array
        y_pred_numpy = self.tensor_to_numpy(y_pred)
        return y_pred_numpy*(self.y_max-self.y_min)+self.y_min
    
    def append_extra_data(self, extra_data):
        self.extra_data = extra_data

    def save_model(self, filepath):
        save_data = {
                'model_state_dict': self.model.state_dict(),
                'X_min': self.X_min,
                'X_max': self.X_max,
                'y_min': self.y_min,
                'y_max': self.y_max,
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'loss_history': self.loss_history,
                'extra_data': self.extra_data,
            }
        torch.save(save_data, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.X_min = checkpoint['X_min']
        self.X_max = checkpoint['X_max']
        self.y_min = checkpoint['y_min']
        self.y_max = checkpoint['y_max']
        self.n_epochs = checkpoint['n_epochs']
        self.batch_size = checkpoint['batch_size']
        self.learning_rate = checkpoint['learning_rate']
        self.loss_history = checkpoint['loss_history']
        self.extra_data = checkpoint['extra_data']

    def PCA_fit(self, X, n_components=32):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        pca.fit(X)
        X_transformed = pca.transform(X)
        print(X.shape, X_transformed.shape, pca.score(X))
        return pca 
    
    def PCA_transform_data(self, data, pca):
        out = pca.transform(data)
        return out

    def PCA_inverse_transform_data(self, data, pca):
        out = pca.inverse_transform(data)
        return out 
    

# SIREN layer definition
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, dropout_prob=0.5):
        super(SirenLayer, self).__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)
        self.w0 = w0

    def forward(self, x):
        y = self.linear(x)
        y = self.dropout(y)  # Apply dropout
        return torch.sin(self.w0 * y)

# SIREN model definition
class SirenModel(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super(SirenModel, self).__init__()

        layers = [SirenLayer(in_features, hidden_features)]
        for _ in range(hidden_layers):
            layers.append(SirenLayer(hidden_features, hidden_features))
        layers.append(SirenLayer(hidden_features, out_features))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    




