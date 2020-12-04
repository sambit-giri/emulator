# Imports
import probflow as pf
import numpy as np
import matplotlib.pyplot as plt
rand = lambda *x: np.random.rand(*x).astype('float32')
randn = lambda *x: np.random.randn(*x).astype('float32')
zscore = lambda x: (x-np.mean(x, axis=0))/np.std(x, axis=0)

# Create the data
N = 1024
x = 10*rand(N, 1)-5
y = np.sin(x)/(1+x*x) + 0.05*randn(N, 1)

# Normalize
x = zscore(x)
y = zscore(y)

# Plot it
plt.plot(x, y, '.')


import tensorflow as tf

class DenseLayer(pf.Module):

    def __init__(self, d_in, d_out):
        self.w = pf.Parameter([d_in, d_out])
        self.b = pf.Parameter([1, d_out])

    def __call__(self, x):
        return x @ self.w() + self.b()

class DenseNetwork(pf.Module):

    def __init__(self, dims):
        Nl = len(dims)-1 #number of layers
        self.layers = [DenseLayer(dims[i], dims[i+1]) for i in range(Nl)]
        self.activations = (Nl-1)*[tf.nn.relu] + [lambda x: x]

    def __call__(self, x):
        for i in range(len(self.activations)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x


class DenseRegression(pf.ContinuousModel):

    def __init__(self, dims):
        self.net = DenseNetwork(dims)
        self.s = pf.ScaleParameter([1, 1])

    def __call__(self, x):
        return pf.Normal(self.net(x), self.s())


model = DenseRegression([1, 32, 32, 1])
model.fit(x, y, epochs=1000, lr=0.02)


# Test points to predict
x_test = np.linspace(min(x), max(x), 101).astype('float32').reshape(-1, 1)

# Predict them!
preds = model.predict(x_test)

# Plot it
plt.plot(x, y, '.', label='Data')
plt.plot(x_test, preds, 'r', label='Predictions')
