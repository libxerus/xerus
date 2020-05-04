import os
DIR = os.path.dirname(os.path.realpath(__file__))
import ctypes; ctypes.cdll.LoadLibrary(os.path.join(DIR, "libxerus_misc.so"))
import ctypes; ctypes.cdll.LoadLibrary(os.path.join(DIR, "libxerus.so"))
del os, DIR, ctypes
from xerus.xerus import *
__version__ = xerus.__version__
__doc__ = xerus.__doc__
