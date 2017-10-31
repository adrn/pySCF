from __future__ import absolute_import

from os import path
from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():

    exts = []

    # malloc
    mac_incl_path = "/usr/include/malloc"

    # Get gala path
    import gala
    gala_base_path = path.split(gala.__file__)[0]
    gala_potential_incl = path.join(gala_base_path, 'potential')

    # acceleration.pyx
    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append(gala_potential_incl)

    cfg['extra_compile_args'].append('--std=gnu99')

    cfg['sources'].append('scf/acceleration.pyx')
    cfg['sources'].append(path.join(gala_potential_incl, 'potential', 'src',
                                    'cpotential.c'))
    cfg['sources'].append(path.join(gala_potential_incl, 'potential', 'builtin',
                                    'builtin_potentials.c'))
    cfg['sources'].append('scf/src/scf.c')
    cfg['sources'].append('scf/src/helpers.c')

    cfg['libraries'] = ['gsl', 'gslcblas']
    exts.append(Extension('scf.acceleration', **cfg))

    # scf.pyx
    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append(gala_potential_incl)

    cfg['extra_compile_args'].append('--std=gnu99')

    cfg['sources'].append('scf/scf.pyx')
    cfg['sources'].append(path.join(gala_potential_incl, 'potential', 'src',
                                    'cpotential.c'))
    cfg['sources'].append(path.join(gala_potential_incl, 'potential', 'builtin',
                                    'builtin_potentials.c'))
    cfg['sources'].append('scf/src/scf.c')
    cfg['sources'].append('scf/src/helpers.c')
    cfg['sources'].append('scf/src/leapfrog.c')

    cfg['libraries'] = ['gsl', 'gslcblas']
    exts.append(Extension('scf.scf', **cfg))

    return exts

def get_package_data():
    return {'scf': ['src/*.h', 'tests/data/*']}
