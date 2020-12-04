import numpy as np

try: import torch
except: print('Install PyTorch.')

try:
	import pyro
	import pyro.contrib.gp as gp
	import pyro.distributions as dist
except:
	print('Install Pyro to use BayesianNN.')

