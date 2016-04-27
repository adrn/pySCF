from __future__ import absolute_import

import os
from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_package_data():
    return {
        _ASTROPY_PACKAGE_NAME_ + '.tests': ['coveragerc']}

def get_extensions():

    # Get gary path
    import gary
    gary_base_path = os.path.split(gary.__file__)[0]
    gary_path = os.path.join(gary_base_path, 'potential')

    extensions = []

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(gary_path)
    cfg['include_dirs'].append('scf')
    cfg['sources'].append('scf/tests/helpers.pyx')
    cfg['sources'].append('scf/src/helpers.c')
    cfg['sources'].append('scf/src/scf.c')
    cfg['libraries'] = ['gsl', 'gslcblas']
    cfg['extra_compile_args'] = ['--std=gnu99']
    extensions.append(Extension('scf.tests.helpers', **cfg))

    return extensions
