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

cdef extern from "src/scf.h":
    void accp_LH(int nbodies, double *xyz, double *mass, int *ibound,
                 int nmax, int lmax, int zeroodd, int zeroeven,
                 double *sinsum, double *cossum,
                 double G, int *firstc,
                 double *dblfact, double *twoalpha, double *anltilde, double *coeflm,
                 int *lmin, int *lskip,
                 double *c1, double *c2, double *c3,
                 double *pot,
                 double *acc) nogil

    void acc_pot(int selfgrav, int nbodies, double *xyz, double *mass, int *ibound,
                 int nmax, int lmax, int zeroodd, int zeroeven,
                 double *sinsum, double *cossum,
                 double G, int *firstc,
                 double *dblfact, double *twoalpha, double *anltilde, double *coeflm,
                 int *lmin, int *lskip,
                 double *c1, double *c2, double *c3,
                 double *pot,
                 double *acc) nogil

    void frame(int iter, int n_recenter,
               int nbodies, double *xyz, double *vxyz, double *mass,
               double *pot) nogil

cdef extern from "src/helpers.h":
    void indexx(int n, double *arrin, int *indx) nogil

def scf():
    # read SCFBI file
    skip = 1
    names = ['m','x','y','z','vx','vy','vz']
    bodies_filename = '/Users/adrian/projects/scf_fortran/src/SCFBI'
    bodies = np.genfromtxt(bodies_filename, dtype=None, names=names,
                           skip_header=skip)

    cdef:
        int N = 128
        double G = 1.
        int firstc = 1

        # turn these into (n_bodies,3) arrays
        double[:,::1] xyz = np.ascontiguousarray(np.vstack([bodies[n] for n in ['x','y','z']]).T[:N])
        double[:,::1] vxyz = np.ascontiguousarray(np.vstack([bodies[n] for n in ['vx','vy','vz']]).T[:N])
        double[::1] mass = np.ones(N) / N
        int[::1] ibound = np.ones(N, dtype=np.int32)

        int n_recenter = 100
        int nmax = 6
        int lmax = 4
        int zeroodd = 0
        int zeroeven = 0
        int external_field = 1
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
        double[::1] xyz_frame = np.array([15.,0,0])
        double[::1] vxyz_frame = np.array([0,0.2,0])

    # this stuff follows `initsys`
    acc_pot(selfgrav, N, &xyz[0,0], &mass[0], &ibound[0],
            nmax, lmax, zeroodd, zeroeven,
            &sinsum[0,0,0], &cossum[0,0,0],
            G, &firstc,
            &dblfact[0], &twoalpha[0], &anltilde[0,0], &coeflm[0,0],
            &lmin, &lskip,
            &c1[0,0], &c2[0,0], &c3[0],
            &pot[0], &acc[0,0])

    if external_field:
        frame(0, n_recenter, N, &xyz[0,0], &vxyz[0,0], &mass[0], &pot[0])

    # initvel(...stuff) # TODO
    # print(np.array(pot[:10]))
