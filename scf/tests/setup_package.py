from os import path
from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_package_data():
    return {
        _ASTROPY_PACKAGE_NAME_ + '.tests': ['coveragerc']}

def get_extensions():

    # To find gsl location
    import cython_gsl

    # Get gala include path
    import gala
    gala_base_path = path.split(gala.__file__)[0]
    gala_potential_incl = path.join(gala_base_path, 'potential')

    # Config stuff needed by all extensions
    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(gala_potential_incl)
    cfg['include_dirs'].append(cython_gsl.get_cython_include_dir())
    cfg['include_dirs'].append('scf')

    cfg['sources'].append(path.join(gala_potential_incl,
                                    'potential', 'src', 'cpotential.c'))
    cfg['sources'].append(path.join(gala_potential_incl,
                                    'potential', 'builtin',
                                    'builtin_potentials.c'))
    cfg['sources'].append('scf/src/helpers.c')

    cfg['libraries'] = cython_gsl.get_libraries()
    cfg['library_dirs'].append(cython_gsl.get_library_dir())
    cfg['extra_compile_args'].append('--std=gnu99')

    # acceleration.pyx
    cfg['sources'].append('scf/tests/helpers.pyx')

    return [Extension('scf.tests.helpers', **cfg)]
