import numpy as np
from time import time 
import pickle
from pyDOE import lhs
from . import helper_functions as hf

def _objective_function(x):
    # Replace this with the actual objective function to be minimized or maximized
    return np.sum(x**2)

def LH_sampling(n_params=2, n_samples=10, mins=0, maxs=1, outfile=None, **kwargs):
	"""
	Latin Hypercube Sampling.

	Parameters:
	-----------
	n_params (int): Give the number of parameters.
	n_samples (int): Total number of points required in the parameter space.
	mins (float or list): Minimum value for the parametrs. If you give a float, then it assumes same minimum value for all the parameters.
	maxs (float or list): Maximum value for the parameters. If you give a float, then it assumes same maximum value for all the parameters.
	outfile (str): Name of the text file where the parameter values will be saved. If None, the values will not be saved anywhere.
	
	Return:
	-------
	An array containing the parameter values.
	"""
	if isinstance(mins,list): mins = np.array(mins)
	if isinstance(maxs,list): maxs = np.array(maxs)
	
	verbose = kwargs.get('verbose', True)
	tstart = time()
	if verbose: print(f'Sampling {len(mins)} dimensional space...')

	#lhs(n, [samples, criterion, iterations])
	lhd = lhs(n_params, samples=n_samples)
	lhd = mins + (maxs-mins)*lhd

	if outfile is not None:	np.savetxt(outfile.split('.txt')[0]+'.txt', lhd)
	if verbose: 
		t_sec = time()-tstart
		print(f'...done in {t_sec:.3f} s' if t_sec>0.001 else f'...done in {t_sec*1e3:.3f} ms')
	return lhd

def MC_sampling(n_params=2, n_samples=10, mins=0, maxs=1, outfile=None, **kwargs):
	"""
	Monte Carlo Sampling.

	Parameters:
	-----------
	n_params (int): Give the number of parameters.
	n_samples (int): Total number of points required in the parameter space.
	mins (float or list): Minimum value for the parametrs. If you give a float, then it assumes same minimum value for all the parameters.
	maxs (float or list): Maximum value for the parameters. If you give a float, then it assumes same maximum value for all the parameters.
	outfile (str): Name of the text file where the parameter values will be saved. If None, the values will not be saved anywhere.
	
	Return:
	-------
	An array containing the parameter values.
	"""
	if isinstance(mins,list): mins = np.array(mins)
	if isinstance(maxs,list): maxs = np.array(maxs)
	
	verbose = kwargs.get('verbose', True)
	tstart = time()
	if verbose: print(f'Sampling {len(mins)} dimensional space...')

	mcd = np.random.uniform(size=(n_samples,n_params))
	for i,[mn,mx] in enumerate(zip(mins,maxs)): mcd[:,i] = mn + (mx-mn)*mcd[:,i]

	if outfile is not None:	np.savetxt(outfile.split('.txt')[0]+'.txt', mcd)
	if verbose: 
		t_sec = time()-tstart
		print(f'...done in {t_sec:.3f} s' if t_sec>0.001 else f'...done in {t_sec*1e3:.3f} ms')
	return mcd

def PLH_sampling(n_params=2, n_samples=10, mins=0, maxs=1, outfile=None, n_slices=100, **kwargs):
	"""
	Progressive Latin Hypercube Sampling (Sheikholeslami & Razavi 2017).

	Parameters:
	-----------
	n_params (int): Give the number of parameters.
	n_samples (int): Total number of points required in the parameter space.
	mins (float or list): Minimum value for the parametrs. If you give a float, then it assumes same minimum value for all the parameters.
	maxs (float or list): Maximum value for the parameters. If you give a float, then it assumes same maximum value for all the parameters.
	outfile (str): Name of the text file where the parameter values will be saved. If None, the values will not be saved anywhere.
	n_slices (int): Give the number of slices or progressions.

	Return:
	-------
	An array containing the parameter values.
	"""
	if isinstance(mins,list): mins = np.array(mins)
	if isinstance(maxs,list): maxs = np.array(maxs)
	
	verbose = kwargs.get('verbose', True)
	tstart = time()
	if verbose: print(f'Sampling {len(mins)} dimensional space...')

	# Step 1: Initialization
	criterion = kwargs.get('criterion')
	initial_sample = lhs(n_params, samples=n_samples, criterion=criterion)

	for iteration in range(n_slices):
		# Step 2: Evaluation
		objective_values = np.apply_along_axis(_objective_function, 1, initial_sample)
		# Step 3: Sorting
		sorted_indices = np.argsort(objective_values)
		ranked_sample = initial_sample[sorted_indices]
		# Step 4: Divide and Refine
		subregions = np.array_split(ranked_sample, n_slices, axis=0)
		refined_sample = np.zeros_like(initial_sample)

		for i, subregion in enumerate(subregions):
			refined_subregion = lhs(n_params, samples=len(subregion), criterion=criterion)
			refined_sample[i::n_slices, :] = refined_subregion
		# Step 5: Combine
		initial_sample = refined_sample

	# Return the final Latin Hypercube sample
	final_sample = initial_sample*(maxs-mins)+mins 
	if outfile is not None:	np.savetxt(outfile.split('.txt')[0]+'.txt', final_sample)
	if verbose: 
		t_sec = time()-tstart
		print(f'...done in {t_sec:.3f} s' if t_sec>0.001 else f'...done in {t_sec*1e3:.3f} ms')
	return final_sample

def SLH_sampling(n_params=2, n_samples=10, mins=0, maxs=1, outfile=None, n_slices=100, **kwargs):
	"""
	Successive Latin Hypercube Sampling.
	IMPLEMENTATION INCOMPLETE.

	Parameters:
	-----------
	n_params (int): Give the number of parameters.
	n_samples (int): Total number of points required in the parameter space.
	mins (float or list): Minimum value for the parametrs. If you give a float, then it assumes same minimum value for all the parameters.
	maxs (float or list): Maximum value for the parameters. If you give a float, then it assumes same maximum value for all the parameters.
	outfile (str): Name of the text file where the parameter values will be saved. If None, the values will not be saved anywhere.
	n_slices (int): Give the number of slices or progressions.

	Return:
	-------
	An array containing the parameter values.
	"""
	if isinstance(mins,list): mins = np.array(mins)
	if isinstance(maxs,list): maxs = np.array(maxs)
	
	verbose = kwargs.get('verbose', True)
	tstart = time()
	if verbose: print(f'Sampling {len(mins)} dimensional space...')

	# Step 1: Initialization
	criterion = kwargs.get('criterion')
	initial_sample = lhs(n_params, samples=n_samples, criterion=criterion)

	# Dictionary to store samples at different iterations
	lhs_samples_dict = {'iteration_0': initial_sample*(maxs-mins)+mins}

	for iteration in range(n_slices):
		# Step 2: Evaluation
		objective_values = np.apply_along_axis(_objective_function, 1, initial_sample)
		# Step 3: Sorting
		sorted_indices = np.argsort(objective_values)
		ranked_sample = initial_sample[sorted_indices]
		# Step 4: Divide and Refine
		subregions = np.array_split(ranked_sample, n_slices, axis=0)
		refined_sample = np.zeros_like(initial_sample)
		for i, subregion in enumerate(subregions):
			refined_subregion = lhs(n_params, samples=len(subregion), criterion='maximin')
			refined_sample[i::n_slices, :] = refined_subregion
		# Step 5: Combine
		initial_sample = refined_sample
		# Store the current sample in the dictionary
		lhs_samples_dict[f'iteration_{iteration + 1}'] = initial_sample*(maxs-mins)+mins

	# Return the dictionary of Latin Hypercube samples at different iterations
	if outfile is not None:	
		outfile = outfile.split('.pkl')[0]+'.pkl'
		pickle.dump(lhs_samples_dict, open(outfile, 'wb'))
	if verbose: 
		t_sec = time()-tstart
		print(f'...done in {t_sec:.3f} s' if t_sec>0.001 else f'...done in {t_sec*1e3:.3f} ms')
	return lhs_samples_dict

def MCS_nsphere(n_params=2, n_samples=10, mins=0, maxs=1, outfile=None, **kwargs):
	"""
	Parameters:
	-----------
	n_params (int): Give the number of parameters.
	n_samples (int): Total number of points required in the parameter space.
	mins (float or list): Minimum value for the parametrs. If you give a float, then it assumes same minimum value for all the parameters.
	maxs (float or list): Maximum value for the parameters. If you give a float, then it assumes same maximum value for all the parameters.
	outfile (str): Name of the text file where the parameter values will be saved. If None, the values will not be saved anywhere.
	
	Return:
	-------
	An array containing the parameter values.
	"""
	if isinstance(mins,list): mins = np.array(mins)
	if isinstance(maxs,list): maxs = np.array(maxs)
	
	verbose = kwargs.get('verbose', True)
	tstart = time()
	if verbose: print(f'Sampling {len(mins)} dimensional space...')

	mcd = np.random.uniform(size=(n_samples,n_params))
	mcd_r = ((mcd-0.5)**2).sum(axis=1)
	mcd = mcd[mcd_r<0.25]
	while mcd.shape[0]<n_samples:
		mcdi = np.random.uniform(size=(1,n_params))
		mcd_ri = ((mcdi-0.5)**2).sum(axis=1)
		if mcd_ri<0.25: mcd = np.vstack((mcd, mcdi))

	#print(mcd.shape)
	for i,[mn,mx] in enumerate(zip(mins,maxs)): mcd[:,i] = mn + (mx-mn)*mcd[:,i]
	if outfile is not None:	np.savetxt(outfile.split('.txt')[0]+'.txt', mcd)

	if verbose: 
		t_sec = time()-tstart
		print(f'...done in {t_sec:.3f} s' if t_sec>0.001 else f'...done in {t_sec*1e3:.3f} ms')
	return mcd

def LHS_nsphere(n_params=2, n_samples=10, mins=0, maxs=1, outfile=None, **kwargs):
	"""
	Parameters:
	-----------
	n_params (int): Give the number of parameters.
	n_samples (int): Total number of points required in the parameter space.
	mins (float or list): Minimum value for the parametrs. If you give a float, then it assumes same minimum value for all the parameters.
	maxs (float or list): Maximum value for the parameters. If you give a float, then it assumes same maximum value for all the parameters.
	outfile (str): Name of the text file where the parameter values will be saved. If None, the values will not be saved anywhere.
	
	Return:
	-------
	An array containing the parameter values.
	"""
	if isinstance(mins,list): mins = np.array(mins)
	if isinstance(maxs,list): maxs = np.array(maxs)
	
	verbose = kwargs.get('verbose', True)
	tstart = time()
	if verbose: print(f'Sampling {len(mins)} dimensional space...')

	mins1, maxs1 = [0], [0.5]
	for i in range(len(mins)-2):
		mins1.append(0)
		maxs1.append(np.pi)
	mins1.append(0)
	maxs1.append(2*np.pi)

	lhd1 = LH_sampling(n_params=n_params, samples=n_samples, 
					mins=mins1, maxs=maxs1, 
					outfile=None, verbose=False)

	lhd = []
	for theta in lhd1: lhd.append(hf.spherical_to_cartesian(theta))
	lhd = np.array(lhd); lhd = (lhd-lhd.min(axis=0))/(lhd.max(axis=0)-lhd.min(axis=0))
	# for i,[mn,mx] in enumerate(zip(mins,maxs)): lhd[:,i] = mn + (mx-mn)*lhd[:,i]
	lhd = mins + (maxs-mins)*lhd

	if outfile is not None:	np.savetxt(outfile.split('.txt')[0]+'.txt', lhd)
	if verbose: 
		t_sec = time()-tstart
		print(f'...done in {t_sec:.3f} s' if t_sec>0.001 else f'...done in {t_sec*1e3:.3f} ms')
	return lhd
	
	
if __name__ == '__main__':
	# Here we run a simple example with two parameters.
	# Two parameters.
	n_params = 2
	n_samples = 20
	mins, maxs = [0,-1], [10,1]
	lhd = LH_sampling(n_params=n_params, n_samples=n_samples, mins=mins, maxs=maxs, outfile='lhs_params')
	
	import matplotlib.pyplot as plt

	plt.scatter(lhd[:,0], lhd[:,1])
	plt.title('See that the horizontal and vertical line \n from any point will not intersect any other point.')
	#plt.grid(True)
	dummy = np.random.randint(samples)
	plt.plot(np.linspace(mins[0]-100, maxs[0]+100,10), np.ones(10)*lhd[dummy,1], '--', c='C1')
	plt.plot(np.ones(10)*lhd[dummy,0], np.linspace(mins[1]-100, maxs[1]+100,10), '--', c='C1')
	plt.xlim(mins[0], maxs[0])
	plt.ylim(mins[1], maxs[1])
	plt.show()
	
	print('You can use the create function above in your code by')
	print("\"import create_LHS\" and then")
	print("\"lhd = create(n_params=n_params, samples=samples, mins=mins, maxs=maxs, outfile='lhs_params')\"")