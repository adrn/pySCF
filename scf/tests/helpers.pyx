# coding: utf-8
# cython: debug=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
from astropy.constants import G, M_sun
import h5py
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython
from cpython.exc cimport PyErr_CheckSignals

# from gary.potential.cpotential cimport _CPotential, valuefunc, gradientfunc
# from gary.potential.cpotential cimport CPotentialWrapper
# from gary.potential.cpotential import CPotentialBase

cdef extern from "src/cpotential.h":
    ctypedef struct CPotential:
        pass

cdef extern from "src/scf.h":
    ctypedef struct Config:
        int n_steps
        double dt
        double t0
        int n_bodies
        int n_recenter
        int n_snapshot
        int n_tidal
        int nmax
        int lmax
        int zeroodd
        int zeroeven
        int selfgravitating
        double ru
        double mu
        double vu
        double tu
        double G

    ctypedef struct Placeholders:
        double *dblfact
        double *twoalpha
        double *anltilde
        double *coeflm
        double *plm
        double *dplm
        double *ultrasp
        double *ultraspt
        double *ultrasp1
        double *sinsum
        double *cossum
        double *c1
        double *c2
        double *c3
        int *lmin
        int *lskip
        double *pot0
        double *kin0
        double *ax0
        double *ay0
        double *az0

    ctypedef struct Bodies:
        double *x
        double *y
        double *z
        double *vx
        double *vy
        double *vz
        double *ax
        double *ay
        double *az
        double *pot
        double *kin
        double *mass
        int *ibound
        double *tub

    ctypedef struct COMFrame:
        double x
        double y
        double z
        double vx
        double vy
        double vz
        int *pot_idx

    void accp_firstc(Config config, Placeholders p) nogil

    void acc_pot(Config config, Bodies b, Placeholders p, COMFrame *f,
                 CPotential *pot, double extern_strength, double *tnow,
                 int *firstc) nogil

    void frame(int iter, Config config, Bodies b, COMFrame *f) nogil

    void step_vel(Config config, Bodies b, double dt,
                  double *tnow, double *tvel) nogil

    void step_pos(Config config, Bodies b, double dt,
                  double *tnow, double *tvel) nogil

    void tidal_start(int iter, Config config, Bodies b, Placeholders p,
                     COMFrame *f, CPotential *pot,
                     double *tnow, double *tpos, double *tvel) nogil

    void check_progenitor(int iter, Config config, Bodies b, Placeholders p,
                          COMFrame *f, CPotential *pot, double *tnow) nogil

cpdef _test_accp_firstc():
    cdef:
        Config config
        Placeholders p

        int nmax = 6
        int lmax = 4

        double[::1] dblfact = np.zeros(lmax+1)
        double[::1] twoalpha = np.zeros(lmax+1)
        double[:,::1] anltilde = np.zeros((nmax+1,lmax+1))
        double[:,::1] coeflm = np.zeros((lmax+1,lmax+1))
        double[:,::1] c1 = np.zeros((nmax+1,lmax+1))
        double[:,::1] c2 = np.zeros((nmax+1,lmax+1))
        double[::1] c3 = np.zeros(nmax+1)

        int lmin = 0
        int lskip = 0

    # minimum set of parameters to initialize
    config.nmax = nmax
    config.lmax = lmax
    config.zeroodd = 0
    config.zeroeven = 0

    # Pointers to a bunch of placeholder arrays
    p.lmin = &lmin
    p.lskip = &lskip
    p.dblfact = &dblfact[0]
    p.twoalpha = &twoalpha[0]
    p.anltilde = &anltilde[0,0]
    p.coeflm = &coeflm[0,0]
    p.c1 = &c1[0,0]
    p.c2 = &c2[0,0]
    p.c3 = &c3[0]

