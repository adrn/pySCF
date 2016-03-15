# Licensed under a 3-clause BSD style license - see PYFITS.rst
from __future__ import absolute_import

from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append('scf/src')
    cfg['sources'].append('scf/core.pyx')
    cfg['sources'].append('scf/src/scf.c')
    cfg['sources'].append('scf/src/indexx.c')
    cfg['sources'].append('scf/src/helpers.c')
    cfg['sources'].append('scf/src/nrutil.c')
    cfg['sources'].append('scf/src/leapfrog.c')
    cfg['libraries'] = ['gsl', 'gslcblas']
    cfg['extra_compile_args'] = ['--std=gnu99']

    return [Extension('scf.core', **cfg)]

def get_package_data():
    return {'scf': ['src/*.h']}
