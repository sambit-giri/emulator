import numpy as np 
import sys

def loading_verbose(string):
	msg = (string)
	sys.stdout.write('\r'+msg)
	sys.stdout.flush()

def spherical_to_cartesian(theta):
	x = np.zeros_like(theta)
	x[0] = theta[0]*np.cos(theta[1])
	for i in range(1,theta.shape[0]-1):
		x[i] = x[i-1]*np.tan(theta[i])*np.cos(theta[i+1])
	x[-1] = x[-2]*np.tan(theta[-1])
	return x

def moving_average(data, window_size):
    '''
    Function to compute moving average
    '''
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size