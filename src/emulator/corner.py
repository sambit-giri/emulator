import numpy as np 
import matplotlib.pyplot as plt
from skimage.filters import gaussian

def credible_limit(zi, level, method='naive'):
	if method=='naive':
		zbins = np.linspace(zi.min(), zi.max(), 2000)
		sm = 0
		#while sm<(1-level/100):
		for i,zb in enumerate(zbins):
			sm = np.sum(zi[zi<zb])/np.sum(zi)
			#print(zb,sm)
			if sm>(1-level/100): break
	return zb

def plot_lfire(lfi, smooth=5, true_values=None, CI=[95]):
	if np.ndim(lfi.thetas)==1:
		fig, axes = plt.subplots(nrows=1, ncols=1) 
		xx, cube  = lfi.thetas, lfi.posterior
		if smooth: cube = gaussian(cube, smooth) 
		axes.plot(xx, cube)
		axes.set_xlabel(lfi.param_names[0])
		plt.show()
		return None
	else:
		N = lfi.thetas.shape[1]
		fig, axes = plt.subplots(nrows=N, ncols=N)
		for i in range(N):
			for j in range(N):
				if j>i: axes[i,j].axis('off')
				elif i==j: 
					plot_1Dmarginal_lfire(lfi, i, ax=axes[i,j], smooth=smooth, true_values=true_values)
					if i+1<N: 
						axes[i,j].set_xlabel('')
						axes[i,j].set_xticks([])
					if j>0:
						axes[i,j].set_yticks([])
				else: 
					im = plot_2Dmarginal_lfire(lfi, i, j, ax=axes[i,j], smooth=smooth, true_values=true_values, CI=CI)
					if i+1<N: 
						axes[i,j].set_xlabel('')
						axes[i,j].set_xticks([])
					if j>0: 
						axes[i,j].set_ylabel('')
						axes[i,j].set_yticks([])

	fig.subplots_adjust(right=0.88)
	cb_ax = fig.add_axes([0.89, 0.1, 0.02, 0.8])
	cbar = fig.colorbar(im, cax=cb_ax)
	plt.show()

def plot_1Dmarginal(thetas, posterior, param_names=None, idx=0, ax=None, bins=100, verbose=False, smooth=False, true_values=None):
	N = thetas.shape[1]
	inds = np.arange(N); inds = np.delete(inds, idx)
	X    = np.array([thetas[:,i] for i in inds])
	X    = np.vstack((thetas[:,idx].reshape(1,-1), X)).T
	y    = posterior
	dm   = [int(np.round(y.shape[0]**(1/X.shape[1]))) for i in range(X.shape[1])] 
	cube = y.reshape(dm)
	if idx!=0: cube = np.swapaxes(cube, 0, idx)
	while(cube.ndim>1):
		if verbose: print('Reducing dimension: {0:} -> {1:}'.format(cube.ndim,cube.ndim-1))
		cube = cube.sum(axis=-1)
	xx = np.unique(X[:,0])
	if smooth: cube = gaussian(cube, smooth) 
	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1)
	ax.plot(xx, (cube-cube.min())/(cube.max()-cube.min()))
	if param_names is not None: ax.set_xlabel(param_names[idx])

def plot_2Dmarginal(thetas, posterior, param_names=None, idx=0, idy=1, ax=None, bins=100, verbose=False, smooth=False, true_values=None, CI=[95]):
	N = thetas.shape[1]
	inds = np.arange(N); inds = np.delete(inds, max([idx,idy])); inds = np.delete(inds, min([idx,idy]))
	X = np.array([thetas[:,i] for i in inds])
	X = np.vstack((thetas[:,idx].reshape(1,-1), thetas[:,idy].reshape(1,-1))).T if X.size==0 else np.vstack((thetas[:,idx].reshape(1,-1), thetas[:,idy].reshape(1,-1), X)).T
	y = posterior
	dm   = [int(np.round(y.shape[0]**(1/X.shape[1]))) for i in range(X.shape[1])] 
	cube = y.reshape(dm)
	cube = np.swapaxes(cube, 0, idx)
	cube = np.swapaxes(cube, 1, idy)
	while(cube.ndim>2):
		if verbose: print('Reducing dimension: {0:} -> {1:}'.format(cube.ndim,cube.ndim-1))
		cube = cube.sum(axis=-1)
	yy = np.unique(X[:,0])
	xx = np.unique(X[:,1])
	xi, yi = np.meshgrid(xx,yy)
	if smooth: cube = gaussian(cube, smooth) 
	if ax is None: fig, ax = plt.subplots(nrows=1, ncols=1)
	zi = (cube-cube.min())/(cube.max()-cube.min())
	im = ax.pcolormesh(xi, yi, zi, cmap='Blues')
	if true_values is not None: 
		ax.scatter(true_values[param_names[idy]], true_values[param_names[idx]], marker='*', c='r')
	if CI is not None:
		for cc in CI:
			ll = credible_limit(zi, cc, method='naive')
			#print(ll)
			ax.contour(xi, yi, zi, levels=[ll], linewidths=0.5, colors='k')
	if ax is None: fig.colorbar(im, ax=ax)
	#ax.imshow(xx, cube)
	if param_names is not None:
		ax.set_xlabel(param_names[idy])
		ax.set_ylabel(param_names[idx])
	return im


def plot_1Dmarginal_lfire(lfi, idx, ax=None, bins=100, verbose=False, smooth=False, true_values=None):
	N = lfi.thetas.shape[1]
	thetas = lfi.thetas
	inds = np.arange(N); inds = np.delete(inds, idx)
	X    = np.array([thetas[:,i] for i in inds])
	X    = np.vstack((thetas[:,idx].reshape(1,-1), X)).T
	y    = lfi.posterior
	dm   = [int(np.round(y.shape[0]**(1/X.shape[1]))) for i in range(X.shape[1])] 
	cube = y.reshape(dm)
	if idx!=0: cube = np.swapaxes(cube, 0, idx)
	while(cube.ndim>1):
		if verbose: print('Reducing dimension: {0:} -> {1:}'.format(cube.ndim,cube.ndim-1))
		cube = cube.sum(axis=-1)
	xx = np.unique(X[:,0])
	if smooth: cube = gaussian(cube, smooth) 
	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1)
	ax.plot(xx, (cube-cube.min())/(cube.max()-cube.min()))
	ax.set_xlabel(lfi.param_names[idx])
	

def plot_2Dmarginal_lfire(lfi, idx, idy, ax=None, bins=100, verbose=False, smooth=False, true_values=None, CI=[95]):
	N = lfi.thetas.shape[1]
	thetas = lfi.thetas
	inds = np.arange(N); inds = np.delete(inds, max([idx,idy])); inds = np.delete(inds, min([idx,idy]))
	X = np.array([thetas[:,i] for i in inds])
	X = np.vstack((thetas[:,idx].reshape(1,-1), thetas[:,idy].reshape(1,-1))).T if X.size==0 else np.vstack((thetas[:,idx].reshape(1,-1), thetas[:,idy].reshape(1,-1), X)).T
	y = lfi.posterior
	dm   = [int(np.round(y.shape[0]**(1/X.shape[1]))) for i in range(X.shape[1])] 
	cube = y.reshape(dm)
	cube = np.swapaxes(cube, 0, idx)
	cube = np.swapaxes(cube, 1, idy)
	while(cube.ndim>2):
		if verbose: print('Reducing dimension: {0:} -> {1:}'.format(cube.ndim,cube.ndim-1))
		cube = cube.sum(axis=-1)
	yy = np.unique(X[:,0])
	xx = np.unique(X[:,1])
	xi, yi = np.meshgrid(xx,yy)
	if smooth: cube = gaussian(cube, smooth) 
	if ax is None: fig, ax = plt.subplots(nrows=1, ncols=1)
	zi = (cube-cube.min())/(cube.max()-cube.min())
	im = ax.pcolormesh(xi, yi, zi, cmap='Blues')
	if true_values is not None: 
		ax.scatter(true_values[lfi.param_names[idy]], true_values[lfi.param_names[idx]], marker='*', c='r')
	if CI is not None:
		for cc in CI:
			ll = credible_limit(zi, cc, method='naive')
			#print(ll)
			ax.contour(xi, yi, zi, levels=[ll], linewidths=0.5, colors='k')
	if ax is None: fig.colorbar(im, ax=ax)
	#ax.imshow(xx, cube)
	ax.set_xlabel(lfi.param_names[idy])
	ax.set_ylabel(lfi.param_names[idx])
	return im


