from __future__ import absolute_import

import os
from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():

    # Get gala path
    import gala
    gala_base_path = os.path.split(gala.__file__)[0]
    gala_path = os.path.join(gala_base_path, 'potential')

    extensions = []

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(gala_path)
    # cfg['include_dirs'].append('scf/src')
    cfg['sources'].append('scf/scf.pyx')
    cfg['sources'].append('scf/src/scf.c')
    cfg['sources'].append('scf/src/helpers.c')
    cfg['sources'].append('scf/src/leapfrog.c')

    # need to include this for some reason
    cfg['sources'].append(os.path.join(gala_path, 'src', 'cpotential.c'))

    cfg['libraries'] = ['gsl', 'gslcblas']
    cfg['extra_compile_args'] = ['--std=gnu99']
    extensions.append(Extension('scf.scf', **cfg))

    return extensions

def get_package_data():
    return {'scf': ['src/*.h', 'tests/data/*']}
