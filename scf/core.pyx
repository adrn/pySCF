# coding: utf-8
# cython: debug=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

cdef extern from "math.h":
    double sqrt(double)

cdef extern from "src/scf.h":
    ctypedef struct Config:
        int n_bodies
        int n_recenter
        int n_snapshot
        int n_tidal
        int nmax
        int lmax
        int zeroodd
        int zeroeven
        double ru
        double mu
        double vu
        double tu

    void acc_pot(Config config, int selfgrav, double extern_strength,
                 double *xyz, double *mass, int *ibound,
                 double *sinsum, double *cossum,
                 double G, int *firstc,
                 double *dblfact, double *twoalpha, double *anltilde, double *coeflm,
                 int *lmin, int *lskip,
                 double *c1, double *c2, double *c3,
                 double *pot,
                 double *acc) nogil

    void frame(Config config, int iter,
               double *xyz, double *vxyz, double *mass, double *pot,
               int *pot_idx, double *xyz_frame, double *vxyz_frame) nogil

    void initvel(Config config, double *tnow, double *tvel, double dt,
                 double *vxyz, double *acc) nogil

cdef extern from "src/helpers.h":
    void indexx(int n, double *arrin, int *indx) nogil

def scf():
    # read SCFBI file
    skip = 1
    names = ['m','x','y','z','vx','vy','vz']
    bodies_filename = '/Users/adrian/projects/scf/fortran/SCFBI'
    bodies = np.genfromtxt(bodies_filename, dtype=None, names=names,
                           skip_header=skip)

    cdef:
        int N = 128
        # int N = 10000
        double G = 1.
        double t0 = 0.
        double dt = 0.1
        int firstc = 1
        Config config

        # turn these into (n_bodies,3) arrays
        double[:,::1] xyz = np.ascontiguousarray(np.vstack([bodies[n] for n in ['x','y','z']]).T[:N])
        double[:,::1] vxyz = np.ascontiguousarray(np.vstack([bodies[n] for n in ['vx','vy','vz']]).T[:N])
        # double[::1] mass = np.ones(N) / N
        double[::1] mass = np.ones(N) * 1E-4 # HACK: so I can compare to SCF
        int[::1] ibound = np.ones(N, dtype=np.int32)

        int nmax = 6
        int lmax = 4

        int selfgrav = 1

        double[:,:,::1] sinsum = np.zeros((nmax+1,lmax+1,lmax+1))
        double[:,:,::1] cossum = np.zeros((nmax+1,lmax+1,lmax+1))

        # These are set automatically, just need to allocate them
        int lmin = 0
        int lskip = 0

        double[::1] dblfact = np.zeros(lmax+1)
        double[::1] twoalpha = np.zeros(lmax+1)
        double[:,::1] anltilde = np.zeros((nmax+1,lmax+1))
        double[:,::1] coeflm = np.zeros((lmax+1,lmax+1))

        double[:,::1] c1 = np.zeros((nmax+1,lmax+1))
        double[:,::1] c2 = np.zeros((nmax+1,lmax+1))
        double[::1] c3 = np.zeros(nmax+1)

        double[:,::1] acc = np.zeros((N,3))
        double[::1] pot = np.zeros(N)

        int i

        # the position and velocity of the progenitor
        double[::1] xyz_frame = np.array([15.,0,0]) # kpc
        double[::1] vxyz_frame = np.array([0,200.,0]) # km/s whaaat?

        # index array for sorting particles on potential value
        int[::1] pot_idx = np.zeros(N, dtype=np.int32)

        # sim units
        double ru = 0.01 # kpc
        double mu = 2.5e5 # msun
        double gee = 6.67e-8 # NOTE: this is wrong
        double msun = 1.989e33
        double cmpkpc = 3.085678e21
        double secperyr = 3.1536e7
        double tu = sqrt((cmpkpc*ru)**3/(msun*mu*gee))
        double vu = (ru*cmpkpc*1.e-5)/tu

        # for turning on the potential:
        double extern_strength

        double tnow, tpos, tvel

    config.ru = ru
    config.mu = mu
    config.tu = tu
    config.vu = vu
    config.n_bodies = N
    config.n_recenter = 100
    config.n_snapshot = 10
    config.n_tidal = 100
    config.nmax = nmax
    config.lmax = lmax
    config.zeroodd = 0
    config.zeroeven = 0

    if config.n_tidal > 0:
        extern_strength = 0.
    else:
        extern_strength = 1.

    # -------------------------------
    # this stuff follows `initsys`
    tnow = t0
    tpos = tnow
    tvel = tnow

    acc_pot(config, selfgrav, extern_strength,
            &xyz[0,0], &mass[0], &ibound[0],
            &sinsum[0,0,0], &cossum[0,0,0],
            G, &firstc,
            &dblfact[0], &twoalpha[0], &anltilde[0,0], &coeflm[0,0],
            &lmin, &lskip,
            &c1[0,0], &c2[0,0], &c3[0],
            &pot[0], &acc[0,0])

    for i in range(3):
        xyz_frame[i] = xyz_frame[i] / ru
        vxyz_frame[i] = vxyz_frame[i] / vu

    for i in range(N):
        vxyz[i,0] = vxyz[i,0] + vxyz_frame[0]
        vxyz[i,1] = vxyz[i,1] + vxyz_frame[1]
        vxyz[i,2] = vxyz[i,2] + vxyz_frame[2]

    frame(config, 0, &xyz[0,0], &vxyz[0,0], &mass[0], &pot[0],
          &pot_idx[0], &xyz_frame[0], &vxyz_frame[0])

    initvel(config, &tnow, &tvel, dt, &vxyz[0,0], &acc[0,0])

    for i in range(4):
        print("xyz", xyz[i,0], xyz[i,1], xyz[i,2])
        print("vxyz", vxyz[i,0], vxyz[i,1], vxyz[i,2])
        print("axyz", acc[i,0], acc[i,1], acc[i,2])
        print()

    # print(xyz_frame[0], xyz_frame[1], xyz_frame[2])

    # initvel(...stuff) # TODO
    # print(np.array(pot[:10]))
