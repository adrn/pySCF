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

cdef extern from "src/helpers.h":
    int getIndex2D(int row, int col, int ncol)
    int getIndex3D(int row, int col, int dep, int ncol, int ndep)

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
    void accp_bfe(Config config, Bodies b, Placeholders p, int *firstc) nogil

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

cpdef wrap_getIndex2D(int row, int col, int ncol):
    return getIndex2D(row, col, ncol)

cpdef wrap_getIndex3D(int row, int col, int dep, int ncol, int ndep):
    return getIndex3D(row, col, dep, ncol, ndep)

cpdef _test_accp_firstc(int nmax, int lmax):
    cdef:
        Config config
        Placeholders p

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

    accp_firstc(config, p)

    return {
        'dblfact': dblfact,
        'twoalpha': twoalpha,
        'anltilde': anltilde,
        'coeflm': coeflm,
        'c1': c1,
        'c2': c2,
        'c3': c3
    }

cpdef _test_accp_bfe(bodies):
    cdef:
        Config config
        Placeholders p
        Bodies b

        int nmax = 6
        int lmax = 4

        # Read in the phase-space positions of the N bodies
        int N = bodies.shape[0]
        double[::1] x = np.ascontiguousarray(bodies.pos.value[0][:])
        double[::1] y = np.ascontiguousarray(bodies.pos.value[1][:])
        double[::1] z = np.ascontiguousarray(bodies.pos.value[2][:])
        double[::1] vx = np.ascontiguousarray(bodies.vel.value[0][:])
        double[::1] vy = np.ascontiguousarray(bodies.vel.value[1][:])
        double[::1] vz = np.ascontiguousarray(bodies.vel.value[2][:])
        double[::1] mass = np.ones(N) / N
        int[::1] ibound = np.ones(N, dtype=np.int32)
        double[::1] tub = np.zeros(N)
        double[::1] ax = np.zeros(N)
        double[::1] ay = np.zeros(N)
        double[::1] az = np.zeros(N)
        double[::1] pot = np.zeros(N) # (internal) potential energy
        double[::1] kin = np.zeros(N) # kinetic energy

        double[::1] dblfact = np.zeros(lmax+1)
        double[::1] twoalpha = np.zeros(lmax+1)
        double[:,::1] anltilde = np.zeros((nmax+1,lmax+1))
        double[:,::1] coeflm = np.zeros((lmax+1,lmax+1))
        double[:,::1] c1 = np.zeros((nmax+1,lmax+1))
        double[:,::1] c2 = np.zeros((nmax+1,lmax+1))
        double[::1] c3 = np.zeros(nmax+1)
        double[:,::1] plm = np.zeros((lmax+1,lmax+1))
        double[:,::1] dplm = np.zeros((lmax+1,lmax+1))
        double[:,::1] ultrasp = np.zeros((nmax+1,lmax+1))
        double[:,::1] ultraspt = np.zeros((nmax+1,lmax+1))
        double[:,::1] ultrasp1 = np.zeros((nmax+1,lmax+1))
        double[:,:,::1] sinsum = np.zeros((nmax+1,lmax+1,lmax+1))
        double[:,:,::1] cossum = np.zeros((nmax+1,lmax+1,lmax+1))

        int lmin = 0
        int lskip = 0

        int firstc = 1

    # minimum set of parameters to initialize
    config.n_bodies = N
    config.nmax = nmax
    config.lmax = lmax
    config.zeroodd = 0
    config.zeroeven = 0

    # The N bodies
    b.x = &x[0]
    b.y = &y[0]
    b.z = &z[0]
    b.vx = &vx[0]
    b.vy = &vy[0]
    b.vz = &vz[0]
    b.ax = &ax[0]
    b.ay = &ay[0]
    b.az = &az[0]
    b.pot = &pot[0]
    b.kin = &kin[0]
    b.mass = &mass[0]
    b.ibound = &ibound[0]
    b.tub = &tub[0]

    # Pointers to a bunch of placeholder arrays
    p.lmin = &lmin
    p.lskip = &lskip
    p.dblfact = &dblfact[0]
    p.twoalpha = &twoalpha[0]
    p.anltilde = &anltilde[0,0]
    p.coeflm = &coeflm[0,0]
    p.plm = &plm[0,0]
    p.dplm = &dplm[0,0]
    p.ultrasp = &ultrasp[0,0]
    p.ultraspt = &ultraspt[0,0]
    p.ultrasp1 = &ultrasp1[0,0]
    p.sinsum = &sinsum[0,0,0]
    p.cossum = &cossum[0,0,0]
    p.c1 = &c1[0,0]
    p.c2 = &c2[0,0]
    p.c3 = &c3[0]

    accp_bfe(config, b, p, &firstc)

    return {
        'sinsum': sinsum,
        'cossum': cossum,
        'plm': plm,
        'ultrasp': ultrasp,
        'ultraspt': ultraspt,
    }
