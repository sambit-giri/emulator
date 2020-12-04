import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from importlib import reload
import psi

yerr_param = [-1, 2.0]#[0.1, 1.5]
line  = psi.sample_models.noisy_line(yerr_param=yerr_param)
xs    = line.xs()
y_obs = line.observation()

#### For BOLFI
simulator = lambda x: line.simulator(x, line.true_intercept)
distance  = psi.distances.euclidean

#### BOLFI
#from pyDOE import *
#lhd = lhs(2, samples=5)

# 1 param
prior  = {'m': 'uniform'}#, 'c': 'uniform'}
bounds = {'m': [-2.5, 0.5]}#, 'c': [0,10]}
gpr = GaussianProcessRegressor()

rn = psi.BOLFI_1param(simulator, distance, y_obs, prior, bounds, N_init=10, gpr=gpr, max_iter=100)
rn.run()

#Plot
plt.subplot(121)
plt.plot(rn.xout, rn.post_mean_normmax[-1])
#plt.plot(rn.successive_JS_dist, c='C0')
plt.subplot(122)
plt.plot(rn.cv_JS_dist['mean'], c='C1')	

# 2 param
simulator = lambda x: line.simulator(x[0], x[1])
distance  = psi.distances.euclidean

prior  = {'m': 'uniform', 'c': 'uniform'}
bounds = {'m': [-2.5, 0.5], 'c': [0,10]}
#gpr = GaussianProcessRegressor()

kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
gpr = GaussianProcessRegressor(kernel=kernel)

rn = psi.BOLFI(simulator, distance, y_obs, prior, bounds, N_init=50, gpr=gpr, max_iter=100, sigma_tol=0.01, successive_JS_tol=0.02, inside_nSphere=False)
#rn = psi.BOLFI_postGPR(simulator, distance, y_obs, prior, bounds, N_init=100, gpr=gpr, max_iter=100, sigma_tol=0.01, successive_JS_tol=0.02)
rn.run()

fig = plt.figure()
ax1 = fig.add_subplot(221)
#ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

psi.corner.plot_2Dmarginal(
    rn.xout,
    rn.post_mean_normmax[-1].flatten(),
    param_names=rn.param_names,
    idx=0,
    idy=1,
    ax=ax3,
    smooth=False,
    true_values={'m': line.true_slope, 'c': line.true_intercept},
    CI=[95],
)

psi.corner.plot_1Dmarginal(
    rn.xout,
    rn.post_mean_normmax[-1].flatten(),
    param_names=None,
    idx=1,
    ax=ax1,
    bins=100,
    verbose=False,
    smooth=False,
    true_values=None,
)

psi.corner.plot_1Dmarginal(
    rn.xout,
    rn.post_mean_normmax[-1].flatten(),
    param_names=None,
    idx=0,
    ax=ax4,
    bins=100,
    verbose=False,
    smooth=3,
    true_values=None,
)
	
## JS over iterations
plt.rcParams['figure.figsize'] = [12, 6]

plt.subplot(121)
#plt.plot(rn.xout, rn.post_mean_normmax[1])
plt.plot(rn.successive_JS_dist, c='C0')
plt.subplot(122)
plt.plot(rn.cv_JS_dist['mean'], c='C1')	

# Plot
plt.rcParams['figure.figsize'] = [12, 6]
plt.subplot(121)
plt.title('Distances')
plt.scatter(rn.params[:,0], rn.params[:,1], c=rn.dists, cmap='jet')
plt.colorbar()
plt.subplot(122)
plt.title('Posterior')
plt.scatter(rn.xout[:,0], rn.xout[:,1], c=rn.post_mean_normmax[-1].flatten(), cmap='Blues')
plt.colorbar()
plt.scatter(line.true_slope,line.true_intercept, marker='*', c='k')

