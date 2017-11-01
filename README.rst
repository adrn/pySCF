pySCF
=====

N-body simulations using the `Self-Consistent Field (SCF)
<http://dx.doi.org/10.1086/171025>`_ method, implemented in Cython+C with an
interface in Python.

**Note: this package is very much in development and there is currently no
stable release**

Documentation
=============

In progress!

Installation
============

Dependencies
------------

The internals of ``pySCF`` are implemented in C and Cython. To build the Cython
code, you'll need to have installed the `GNU scientific library
(GSL) <http://www.gnu.org/software/gsl/>`_. On a Mac, I recommend installing
this with `anaconda <http://anaconda.org>`_ or `Homebrew <http://brew.sh/>`_.
With anaconda, you can install GSL with::

    conda -c https://anaconda.org/asmeurer/gsl install gsl

To build and install ``pySCF``, you'll also need (at minimum):

    - `cython <https://github.com/cython/cython>`_
    - `cython_gsl <https://github.com/twiecki/CythonGSL>`_
    - `numpy <https://github.com/numpy/numpy>`_
    - `astropy <https://github.com/astropy/astropy>`_
    - `gala <https://github.com/adrn/gala>`_

For `conda <http://anaconda.org>`_ users, see the `environment.yml
<https://github.com/adrn/scf/blob/master/environment.yml>`_ file for a list of
dependencies.

For ``pip`` users, see the `pip-requirements.txt
<https://github.com/adrn/scf/blob/master/pip-requirements.txt>`_ file.

Building the source
-------------------

Use the standard::

    python setup.py install

License
=======

MIT: see the `license file <https://github.com/adrn/scf/blob/master/LICENSE>`_.
