# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys

__minimum_python_version__ = "3.5"

class UnsupportedPythonError(Exception):
    pass


minversion = tuple((int(val) for val in __minimum_python_version__.split('.')))
if sys.version_info < minversion:
    raise UnsupportedPythonError("packagename does not support Python < "
                                 "{}".format(__minimum_python_version__))

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    # from example_mod import *
    from .scf import run_scf
    from .core import SCFSimulation
