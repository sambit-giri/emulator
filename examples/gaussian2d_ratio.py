import numpy as np
from importlib import reload
import psi

class gaussian2d_signal:
	def __init__(self, true_mean, true_cov, N=20):
		self.true_mean = true_mean
		self.true_cov  = true_cov
		self.N         = N

	def observation(self):
		return self.simulator(self.true_mean, self.true_cov)
		
	def simulator(self, mean, cov):
		return np.random.multivariate_normal(mean, cov, self.N)
	

true_cov  = [[0.5, 0], [0, 0.5]]	
true_mean = [2,2]

gs = gaussian2d_signal(true_mean, true_cov, N=50)
y_obs = gs.observation()

## LFIRE
# 2 param
simulator2 = lambda x: gs.simulator([x[0], x[1]], true_cov)

prior2  = {'$x_0$': 'uniform', '$y_0$': 'uniform'}
bounds2 = {'$x_0$': [0, 5], '$y_0$': [0,5]}

lfi2 = psi.LFIRE(simulator2, y_obs, prior2, bounds2, sim_out_den=None, n_m=100, n_theta=100, n_grid_out=100, thetas=None)
lfi2.run()

psi.corner.plot_lfire(lfi2)
