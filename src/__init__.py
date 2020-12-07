'''
Emulator is a Python package for constructing emulators.

You can also get documentation for all routines directory from
the interpreter using Python's built-in help() function.
For example:
>>> import emulator as emul
>>> help(emul.GPRemul)
'''

import sys

from . import sample_models
from . import distances
from . import bayesian_optimisation
from . import corner
from . import sampling_space
from .gaussian_process import *
from .neural_networks import *
from .sim2emul import *

#Suppress warnings from zero-divisions and nans
import numpy
numpy.seterr(all='ignore')
