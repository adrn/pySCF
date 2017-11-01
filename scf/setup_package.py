from copy import deepcopy
from os import path
from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():

    exts = []

    # To find gsl location
    import cython_gsl

    # Get gala include path
    import gala
    gala_base_path = path.split(gala.__file__)[0]
    gala_potential_incl = path.join(gala_base_path, 'potential')

    # Config stuff needed by all extensions
    base_cfg = setup_helpers.DistutilsExtensionArgs()
    base_cfg['include_dirs'].append('numpy')
    base_cfg['include_dirs'].append(gala_potential_incl)
    base_cfg['include_dirs'].append(cython_gsl.get_cython_include_dir())

    base_cfg['sources'].append(path.join(gala_potential_incl,
                                         'potential', 'src', 'cpotential.c'))
    base_cfg['sources'].append(path.join(gala_potential_incl,
                                         'potential', 'builtin',
                                         'builtin_potentials.c'))
    base_cfg['sources'].append('scf/src/helpers.c')

    base_cfg['libraries'] = cython_gsl.get_libraries()
    base_cfg['library_dirs'].append(cython_gsl.get_library_dir())
    base_cfg['extra_compile_args'].append('--std=gnu99')

    # acceleration.pyx
    # cfg = {(k,v) for k,v in base_cfg.items()}
    cfg = deepcopy(dict(base_cfg))
    cfg['sources'].append('scf/acceleration.pyx')
    exts.append(Extension('scf.acceleration', **cfg))

    # progenitor.pyx
    cfg = deepcopy(dict(base_cfg))
    cfg['sources'].append('scf/progenitor.pyx')
    # cfg['sources'].append('scf/src/leapfrog.c')
    exts.append(Extension('scf.progenitor', **cfg))

    # scf.pyx
    cfg = deepcopy(dict(base_cfg))
    cfg['sources'].append('scf/scf.pyx')
    # cfg['sources'].append('scf/src/leapfrog.c')
    exts.append(Extension('scf.scf', **cfg))

    return exts

def get_package_data():
    return {'scf': ['src/*.h', 'tests/data/*', '*.pxd']}
