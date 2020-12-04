import numpy as np

class gaussian_signal:
	def __init__(self, true_mean, true_sigma):
		self.true_mean  = true_mean
		self.true_sigma = true_sigma

	def observation(self, N=20):
		return np.random.normal(loc=self.true_mean, scale=self.true_sigma, size=N)

	def simulator(self, mean, sigma, N=1):
		return np.random.normal(loc=mean, scale=sigma, size=N)

class noisy_line:
    def __init__(self, true_slope=-0.9594, true_intercept=4.294, Nx=50, yerr_param=[0.1, 0.5]):
        self.true_slope = true_slope
        self.true_intercept = true_intercept
        self.x = np.sort(10*np.random.rand(Nx))
        self.yerr = lambda n: yerr_param[0]+yerr_param[1]*np.random.rand(n)

    def xs(self):
        return self.x

    def observation(self):
        m, c = self.true_slope, self.true_intercept
        y = lambda x: m*x+c
        ys = y(self.x) + self.yerr(self.x.size)
        return ys

    def simulator(self, slope, intercept):
        m, c = slope, intercept
        y = lambda x: m*x+c
        ys = y(self.x) + self.yerr(self.x.size)
        return ys



