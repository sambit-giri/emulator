import numpy as np 
import matplotlib.pyplot as plt 
from emulator import Fit_GPR

# 1D function

print('First example: Learning a smooth 1D function.')
print('Sinc function')
print('---------------------------------------------')

def func1(x):
	return np.sinc(x)

x1 = np.linspace(0,5,100)
y1 = func1(x1)

n_samples1  = 10
range_param = {'0': [0,5]}
mod1 = Fit_GPR(func1, n_samples1, range_param=range_param, sampling_method='lh-maximin')
mod1.model_func()

plt.figure(figsize=(14,5))

plt.subplot(121); plt.title('With 10 points, $r^2$ score={:.3f}'.format(mod1.r2_score))
plt.scatter(mod1.X_train, mod1.y_train, c='C0', label='training points')
plt.plot(x1, y1, c='k', label='true', lw=3)
plt.plot(x1, mod1.modelled_func(x1), c='r', label='modelled', lw=2.5, ls='--')
plt.legend()

n_samples1  = 30
range_param = {'0': [0,5]}
mod1 = Fit_GPR(func1, n_samples1, range_param=range_param, sampling_method='lh-maximin')
mod1.model_func()

plt.subplot(122); plt.title('With 30 points, $r^2$ score={:.3f}'.format(mod1.r2_score))
plt.scatter(mod1.X_train, mod1.y_train, c='C0', label='training points')
plt.plot(x1, y1, c='k', label='true', lw=3)
plt.plot(x1, mod1.modelled_func(x1), c='r', label='modelled', lw=2.5, ls='--')
plt.legend()

plt.tight_layout()
plt.show()


print('Second example: Learning a sharp 1D function.')
print('Pulse')
print('---------------------------------------------')

def func2(x):
	x_ = np.floor(x.squeeze()/1.5)
	out = np.zeros(x_.shape[0])
	out[x_%2==0] = 1
	return out

x2 = np.linspace(0,5,100)
y2 = func2(x2)

n_samples2  = 10
range_param = {'0': [0,5]}
mod2 = Fit_GPR(func2, n_samples2, range_param=range_param, sampling_method='lh-maximin')
mod2.model_func()

plt.figure(figsize=(14,5))

plt.subplot(121); plt.title('With 10 points, $r^2$ score={:.3f}'.format(mod2.r2_score))
plt.scatter(mod2.X_train, mod2.y_train, c='C0', label='training points')
plt.plot(x2, y2, c='k', label='true', lw=3)
plt.plot(x2, mod2.modelled_func(x2), c='r', label='modelled', lw=2.5, ls='--')
plt.legend()

n_samples2  = 30
range_param = {'0': [0,5]}
mod2 = Fit_GPR(func2, n_samples2, range_param=range_param, sampling_method='lh-maximin')
mod2.model_func()

plt.subplot(122); plt.title('With 30 points, $r^2$ score={:.3f}'.format(mod2.r2_score))
plt.scatter(mod2.X_train, mod2.y_train, c='C0', label='training points')
plt.plot(x2, y2, c='k', label='true', lw=3)
plt.plot(x2, mod2.modelled_func(x2), c='r', label='modelled', lw=2.5, ls='--')
plt.legend()

plt.tight_layout()
plt.show()

# 2D function

print('Third example: Learning a smooth 2D function.')
print('Superimposition of two perpendicular sin waves')
print('----------------------------------------------')

def func3(x):
	assert x.shape[1]==2
	out = np.cos(x[:,0]*2)+np.cos(x[:,1]*2)
	return out

x3_mesh = np.meshgrid(np.linspace(0,5,100),np.linspace(0,5,100))
x3 = np.array([[i,j] for i,j in zip(x3_mesh[0].flatten(),x3_mesh[1].flatten())])
y3 = func3(x3)

plt.figure(figsize=(16,10))

plt.subplot(221); plt.title('Truth')
plt.pcolormesh(x3_mesh[0], x3_mesh[1], y3.reshape(x3_mesh[0].shape)) 
plt.colorbar()

n_samples3  = 20
range_param = {'0': [0,5], '1': [0,5]}
mod3 = Fit_GPR(func3, n_samples3, range_param=range_param, sampling_method='lh-maximin')
mod3.model_func()

plt.subplot(222); plt.title('{} points, $r^2$ score={:.3f}'.format(n_samples3,mod3.r2_score))
plt.pcolormesh(x3_mesh[0], x3_mesh[1], mod3.modelled_func(x3).reshape(x3_mesh[0].shape)) 
plt.scatter(mod3.X_train[:,0], mod3.X_train[:,1], c=mod3.y_train, label='training points', edgecolors='k')
plt.colorbar()

n_samples3  = 50
range_param = {'0': [0,5], '1': [0,5]}
mod3 = Fit_GPR(func3, n_samples3, range_param=range_param, sampling_method='lh-maximin')
mod3.model_func()

plt.subplot(223); plt.title('{} points, $r^2$ score={:.3f}'.format(n_samples3,mod3.r2_score))
plt.pcolormesh(x3_mesh[0], x3_mesh[1], mod3.modelled_func(x3).reshape(x3_mesh[0].shape)) 
plt.scatter(mod3.X_train[:,0], mod3.X_train[:,1], c=mod3.y_train, label='training points', edgecolors='k')
plt.colorbar()

n_samples3  = 100
range_param = {'0': [0,5], '1': [0,5]}
mod3 = Fit_GPR(func3, n_samples3, range_param=range_param, sampling_method='lh-maximin')
mod3.model_func()

plt.subplot(224); plt.title('{} points, $r^2$ score={:.3f}'.format(n_samples3,mod3.r2_score))
plt.pcolormesh(x3_mesh[0], x3_mesh[1], mod3.modelled_func(x3).reshape(x3_mesh[0].shape)) 
plt.scatter(mod3.X_train[:,0], mod3.X_train[:,1], c=mod3.y_train, label='training points', edgecolors='k')
plt.colorbar()

plt.tight_layout()
plt.show()




