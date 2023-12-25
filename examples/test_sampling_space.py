import numpy as np 
import matplotlib.pyplot as plt 
from emulator.sampling_space import *

n_params = 2
n_samples = 20
mins, maxs = [0,-1], [10,1]

lhd1 = LH_sampling(n_params=n_params, n_samples=n_samples, 
                  mins=mins, maxs=maxs, 
                  # outfile='lhs_params',
                  )

lhd2 = LHS_nsphere(n_params=n_params, n_samples=n_samples, 
                  mins=mins, maxs=maxs, 
                  # outfile='lhs_params',
                  )

n_slices = 10
lhd3 = PLH_sampling(n_params=n_params, n_samples=n_samples, 
                  mins=mins, maxs=maxs, 
                  # outfile='lhs_params',
                  n_slices=n_slices,
                  )

fig, axs = plt.subplots(1,3,figsize=(13,5))
fig.suptitle('See that the horizontal and vertical line \n from any point will not intersect any other point.')
axs[0].set_title('LHS - ncube')
axs[0].scatter(lhd1[:,0], lhd1[:,1])
dummy = lhd1[np.random.randint(n_samples)]
axs[0].axvline(dummy[0], ls='--', color='k')
axs[0].axhline(dummy[1], ls='--', color='k')
axs[1].set_title('LHS - nsphere')
axs[1].scatter(lhd2[:,0], lhd2[:,1])
dummy = lhd2[np.random.randint(n_samples)]
axs[1].axvline(dummy[0], ls='--', color='k')
axs[1].axhline(dummy[1], ls='--', color='k')
axs[2].set_title('PLHS - ncube')
axs[2].scatter(lhd3[:,0], lhd3[:,1])
dummy = lhd3[np.random.randint(n_samples)]
axs[2].axvline(dummy[0], ls='--', color='k')
axs[2].axhline(dummy[1], ls='--', color='k')
for ax in axs:
    ax.axis([mins[0], maxs[0], mins[1], maxs[1]])
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
plt.tight_layout()
plt.show()
