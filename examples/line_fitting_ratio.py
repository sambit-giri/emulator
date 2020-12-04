import numpy as np
from importlib import reload
import psi

yerr_param = [-1, 2.0]#[0.1, 1.5]
line  = psi.sample_models.noisy_line(yerr_param=yerr_param)
xs    = line.xs()
y_obs = line.observation()

## LFIRE
# 1 param
simulator = lambda x: line.simulator(x, line.true_intercept)
#distance  = psi.distances.euclidean

prior  = {'m': 'uniform'}#, 'c': 'uniform'}
bounds = {'m': [-2.5, 0.5]}#, 'c': [0,10]}

lfi = psi.LFIRE(simulator, y_obs, prior, bounds, sim_out_den=None, n_m=100, n_theta=100, n_grid_out=100, thetas=None)
lfi.run()

psi.corner.plot_lfire(lfi)

# 2 param
simulator2 = lambda x: line.simulator(x[0], x[1])
#distance2  = psi.distances.euclidean

prior2  = {'m': 'uniform', 'c': 'uniform'}
bounds2 = {'m': [-2.5, 0.5], 'c': [0,10]}

lfi2 = psi.LFIRE(simulator2, y_obs, prior2, bounds2, sim_out_den=None, n_m=100, n_theta=100, n_grid_out=100, thetas=None)
lfi2.run()

true_values = {'m': line.true_slope, 'c': line.true_intercept}
psi.corner.plot_lfire(lfi2)


## LFIRE BO
# 1 param
simulator = lambda x: line.simulator(x, line.true_intercept)
#distance  = psi.distances.euclidean

prior  = {'m': 'uniform'}#, 'c': 'uniform'}
bounds = {'m': [-2.5, 0.5]}#, 'c': [0,10]}

lfi = psi.LFIRE_BayesianOpt(simulator, y_obs, prior, bounds, exploitation_exploration=None, sigma_tol=0.01, sim_out_den=None, n_m=100, n_theta=100, n_grid_out=100, n_init=10, max_iter=1000, tol=1e-4)
lfi.run()

psi.corner.plot_lfire(lfi)

# 2 param
simulator2 = lambda x: line.simulator(x[0], x[1])
#distance2  = psi.distances.euclidean

prior2  = {'m': 'uniform', 'c': 'uniform'}
bounds2 = {'m': [-2.5, 0.5], 'c': [0,10]}

lfi2 = psi.LFIRE_BayesianOpt(simulator2, y_obs, prior2, bounds2, exploitation_exploration=5, sim_out_den=None, n_m=100, n_theta=100, n_grid_out=100, n_init=10, max_iter=1000, tol=1e-4)
lfi2.run()

psi.corner.plot_lfire(lfi2)


## Nuisance parameters

# 2 param
simulator2 = lambda x: line.simulator(x[0], x[1])
#distance2  = psi.distances.euclidean

prior2  = {'m': 'uniform', 'c': 'uniform'}
bounds2 = {'m': [-2.5, 0.5], 'c': [0,10]}

lfi2 = psi.LFIRE_TrainingSetAuto(simulator2, y_obs, prior2, bounds2, n_init=10, n_step=1, n_max=100, n_grid_out=5, thetas=None, verbose=False, penalty='l1', n_jobs=4, clfy=None, lfire=None)
lfi2.run()



