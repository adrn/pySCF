from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
from astropy.utils.data import get_pkg_data_filename
from astropy.io import ascii
import astropy.units as u
import numpy as np
import six

import gary.dynamics as gd

# Project
from .helpers import _test_accp_firstc, _test_accp_bfe

def test_accp_firstc():
    split_pattr = "# -----------------------------------------------------------------------------"

    # don't change these
    nmax = 6
    lmax = 4

    # from Cython implementation
    d = _test_accp_firstc(nmax, lmax)

    # load "truth" from Fortran
    filename = get_pkg_data_filename('data/accp_firstc.txt')

    with open(filename,'r') as f:
        chunks = f.read().split(split_pattr)

    for chunk in chunks:
        tbl = ascii.read(chunk)
        name = tbl.colnames[-1]

        if len(tbl.colnames) == 2:
            for i,val in tbl:
                print(name, i)
                assert np.allclose(d[name][i], val)

        elif len(tbl.colnames) == 3:
            for i,j,val in tbl:
                print(name, i, j)
                assert np.allclose(d[name][i,j], val)

def test_accp_bfe():
    skip = 1
    names = ['m','x','y','z','vx','vy','vz']
    bodies_filename = get_pkg_data_filename('data/SCFBI')
    bodies = np.genfromtxt(bodies_filename, dtype=None, names=names,
                           skip_header=skip)

    b = gd.CartesianPhaseSpacePosition(pos=bodies[['x','y','z']].view(np.float64).reshape(-1,3).T,
                                       vel=bodies[['vx','vy','vz']].view(np.float64).reshape(-1,3).T)

    d = _test_accp_bfe(b)

    # plm - load "truth" from Fortran
    f77_tbl = ascii.read(get_pkg_data_filename('data/plm.txt'))
    for l,m,plm in f77_tbl:
        assert np.allclose(plm,d['plm'][l,m])

    # sinsum and cossum - load "truth" from Fortran
    f77_tbl = ascii.read(get_pkg_data_filename('data/sincossum.txt'))
    for n,l,m,sinsum,cossum in f77_tbl:
        assert np.allclose(sinsum, d['sinsum'][n,l,m])
        assert np.allclose(cossum, d['cossum'][n,l,m])

    # ultrasp and ultraspt - load "truth" from Fortran
    f77_tbl = ascii.read(get_pkg_data_filename('data/ultrasp.txt'))
    for n,l,ultrasp,ultraspt in f77_tbl:
        assert np.allclose(ultrasp, d['ultrasp'][n,l])
        assert np.allclose(ultraspt, d['ultraspt'][n,l])

    # acceleration and potential at position of bodies
    f77_tbl = ascii.read(get_pkg_data_filename('data/accp_bfe.txt'))
    for n,f77_row in enumerate(f77_tbl):
        np.allclose(np.array(list(f77_row)),
                    [d['ax'][n], d['ay'][n], d['az'][n], d['pot'][n]])
