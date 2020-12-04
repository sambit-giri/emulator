import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from . import sampling_space as smp 

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def propose_location_highDimY(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25, xi=1):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    It maximises the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        bounds: The lower and upper bounds of your X space.
        n_restarts: Number of times to restart minimser from a different random start point.
        xi : Tuning parameter, such as Exploitation-exploration trade-off parameter.

    Returns:
        Location of the acquisition function maximum.
    '''
    min_xs = np.zeros((Y_sample.shape[1],X_sample.shape[1]))

    for i in range(min_xs.shape[0]):
        dim = X_sample.shape[1]
        min_val = []#2*Y_sample[:,i].max()#1
        min_x = None
    
        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr, xi=xi)[0,i]
    
        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')    
            if res.fun < min_val:
                min_val = res.fun#[0]
                min_x = res.x           
        min_xs[i,:] = min_x
    return min_x.reshape(-1, 1)

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25, xi=1, inside_nsphere=True, batch=1):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    It maximises the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        bounds: The lower and upper bounds of your X space.
        n_restarts: Number of times to restart minimser from a different random start point.
        xi : Tuning parameter, such as Exploitation-exploration trade-off parameter.

    Returns:
        Location of the acquisition function maximum.
    '''
    if n_restarts<5*batch:
        print('(n_restarts) parameter is changed to 5x(batch)')
        n_restarts = 5*batch

    dim = X_sample.shape[1]
    min_val = 2*Y_sample.max()#1
    min_x = None

    min_vals, min_xs = [], []

    bound_min = bounds.min(axis=1)
    bound_max = bounds.max(axis=1)
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        if inside_nsphere:
            check = check_inside_nsphere(X, bound_min, bound_max)
            if not check: 
                #print(X, 'check failed')
                return np.inf
        #print(X, 'check passed')
        val = -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr, xi=xi)
        return val.reshape(1) if val.size==1 else val
    
    # Find the best optimum by starting from n_restart different random points.
    if inside_nsphere: start_points = smp.MCS_nsphere(n_params=dim, samples=n_restarts, mins=bound_min, maxs=bound_max)
    else: start_points = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim))
    for x0 in start_points:
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B') 
        if check_inside_nsphere(res.x, bound_min, bound_max):
            min_vals.append(res.fun[0])
            min_xs.append(res.x)       
        # if res.fun < min_val:
        #     min_val = res.fun[0]
        #     min_x = res.x   
    args = _argmin(np.array(min_vals), count=batch)        
    min_val = np.array(min_vals)[args]
    min_x   = np.array(min_xs)[args]
    min_x   = min_x.T if batch>1 else min_x.reshape(-1, 1)
    return min_x

def _argmin(x, count=1, axis=None):
    if count==1: return np.argmin(x, axis=axis)
    args = np.argsort(x, axis=axis)
    return args[:count]


def check_inside_nsphere(X, bound_min, bound_max):
    check = np.sum(((X-bound_min)/(bound_max-bound_min)-0.5)**2, axis=1) if X.ndim>1 else np.sum(((X-bound_min)/(bound_max-bound_min)-0.5)**2)
    return check<0.25


def GP_UCB_posterior_space(X, X_sample, Y_sample, gpr, xi=100):
    '''
    Computes the Upper Confidence Bound (UCB) at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    With this acquisition function, we want to find the space maximises.
    
    Args:
        X: Points at which UCB shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    ucb = mu + xi*sigma
    #ucb = xi*sigma

    return ucb

def negativeGP_LCB(X, X_sample, Y_sample, gpr, xi=100):
    '''
    Computes the Lower Confidence Bound (UCB) at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    With this acquisition function, we want to find the space maximises.
    
    Args:
        X: Points at which UCB shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    lcb = mu - xi*sigma
    #lcb = xi*sigma

    return -lcb

def negativeGP_LCB_definedmu(X, X_sample, Y_sample, gpr, mu=0, xi=100):
    '''
    Computes the Lower Confidence Bound (UCB) at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    With this acquisition function, we want to find the space maximises.
    
    Args:
        X: Points at which UCB shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu1, sigma = gpr.predict(X, return_std=True)
    #mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    lcb = mu - xi*sigma
    #lcb = xi*sigma

    return -lcb

