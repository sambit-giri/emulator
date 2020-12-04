'''
Emulate21cm is a Python package for emulation of 21 cm observables.

You can also get documentation for all routines directory from
the interpreter using Python's built-in help() function.
For example:
>>> import tools21cm as t2c
>>> help(t2c.calc_dt)
'''

import sys

from . import sample_models
from . import distances
from . import bayesian_optimisation
from . import corner
from . import sampling_space
from . import gaussian_process
from . import neural_networks
from .emulate import *

#Suppress warnings from zero-divisions and nans
import numpy
numpy.seterr(all='ignore')
