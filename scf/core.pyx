# coding: utf-8
# cython: debug=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython
from cpython.exc cimport PyErr_CheckSignals

cdef extern from "math.h":
    double sqrt(double)

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

    void acc_pot(Config config, Bodies b, Placeholders p, COMFrame *f,
                 double extern_strength, int *firstc) nogil

    void frame(int iter, Config config, Bodies b, COMFrame *f) nogil

    void step_vel(Config config, Bodies b, double dt,
                  double *tnow, double *tvel) nogil

    void step_pos(Config config, Bodies b, double dt,
                  double *tnow, double *tvel) nogil

    void tidal_start(Config config, Bodies b, Placeholders p, COMFrame *f,
                     double *tnow, double *tpos, double *tvel) nogil

    void step_system(int iter, Config config, Bodies b, Placeholders p, COMFrame *f,
                     double *tnow, double *tpos, double *tvel) nogil

cdef extern from "src/helpers.h":
    void indexx(int n, double *arrin, int *indx) nogil

def scf():
    # read SCFBI file
    skip = 1
    names = ['m','x','y','z','vx','vy','vz']
    bodies_filename = '/Users/adrian/projects/scf/fortran/SCFBI'
    bodies = np.genfromtxt(bodies_filename, dtype=None, names=names,
                           skip_header=skip)
    output_path = '/Users/adrian/projects/scf/test/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    cdef:
        # int N = 128
        int N = 10000
        int firstc = 1
        Config config
        Placeholders p
        Bodies b
        COMFrame f

        # Read in the phase-space positions of the N bodies
        double[::1] x = np.ascontiguousarray(bodies['x'][:N])
        double[::1] y = np.ascontiguousarray(bodies['y'][:N])
        double[::1] z = np.ascontiguousarray(bodies['z'][:N])
        double[::1] vx = np.ascontiguousarray(bodies['vx'][:N])
        double[::1] vy = np.ascontiguousarray(bodies['vy'][:N])
        double[::1] vz = np.ascontiguousarray(bodies['vz'][:N])
        double[::1] mass = np.ones(N) / N
        int[::1] ibound = np.ones(N, dtype=np.int32)
        double[::1] tub = np.zeros(N)
        double[::1] ax = np.zeros(N)
        double[::1] ay = np.zeros(N)
        double[::1] az = np.zeros(N)
        double[::1] pot = np.zeros(N) # (internal) potential energy
        double[::1] kin = np.zeros(N) # kinetic energy

        int nmax = 6
        int lmax = 4

        # placeholder arrays, defined once
        double[::1] dblfact = np.zeros(lmax+1)
        double[::1] twoalpha = np.zeros(lmax+1)
        double[:,::1] anltilde = np.zeros((nmax+1,lmax+1))
        double[:,::1] coeflm = np.zeros((lmax+1,lmax+1))
        double[:,::1] plm = np.zeros((lmax+1,lmax+1))
        double[:,::1] dplm = np.zeros((lmax+1,lmax+1))
        double[:,::1] ultrasp = np.zeros((nmax+1,lmax+1))
        double[:,::1] ultraspt = np.zeros((nmax+1,lmax+1))
        double[:,::1] ultrasp1 = np.zeros((nmax+1,lmax+1))
        double[:,:,::1] sinsum = np.zeros((nmax+1,lmax+1,lmax+1))
        double[:,:,::1] cossum = np.zeros((nmax+1,lmax+1,lmax+1))
        double[:,::1] c1 = np.zeros((nmax+1,lmax+1))
        double[:,::1] c2 = np.zeros((nmax+1,lmax+1))
        double[::1] c3 = np.zeros(nmax+1)
        int lmin = 0
        int lskip = 0

        int i,j

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

    # Configuration stuff
    config.n_bodies = N
    config.n_steps = 4096
    config.dt = 1.
    config.t0 = 0.
    config.n_recenter = 100
    config.n_snapshot = 512
    config.n_tidal = 128 # should be >= 1, matched to fortran
    config.selfgravitating = 1
    config.nmax = nmax
    config.lmax = lmax
    config.zeroodd = 0
    config.zeroeven = 0
    config.G = 1.
    config.ru = ru
    config.mu = mu
    config.tu = tu
    config.vu = vu

    # the position and velocity of the progenitor
    f.x = 15.
    f.y = 0.
    f.z = 0.
    f.vx = 0.
    f.vy = 100. # WTF - why km/s???
    f.vz = 0.
    f.pot_idx = &pot_idx[0]

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

    # initial strength of tidal field
    if config.n_tidal > 0:
        extern_strength = 0.
    else:
        extern_strength = 1.

    # -------------------------------
    # this stuff follows `initsys`
    tnow = config.t0
    tpos = tnow
    tvel = tnow

    # TODO: why??
    f.x = f.x / ru
    f.y = f.y / ru
    f.z = f.z / ru
    f.vx = f.vx / vu
    f.vy = f.vy / vu
    f.vz = f.vz / vu

    for i in range(N):
        kin[i] = 0.5 * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
        vx[i] = vx[i] + f.vx
        vy[i] = vy[i] + f.vy
        vz[i] = vz[i] + f.vz

    acc_pot(config, b, p, &f, extern_strength, &firstc)

    frame(0, config, b, &f)

    # initialize velocities (take a half step in time)
    step_vel(config, b, 0.5*config.dt, &tnow, &tvel)

    # for i in range(4):
    #     print("xyz", x[i], y[i], z[i])
    #     # print("pot kin", pot[i], kin[i])
    #     print("vxyz", vx[i], vy[i], vz[i])
    #     print("axyz", ax[i], ay[i], az[i])
    #     print()
    # return

    # slowly turn on tidal field
    # TODO: do tidal start in Cython?
    tidal_start(config, b, p, &f, &tnow, &tpos, &tvel)

    j = 0
    for i in range(config.n_steps):
        PyErr_CheckSignals()
        step_system(i, config, b, p, &f, &tnow, &tpos, &tvel)

        if ((i+1) % config.n_snapshot) == 0 or i == 0:
            snap_filename = os.path.join(output_path, "SNAP{:04d}".format(j))

            step_vel(config, b, 0.5*config.dt, &tnow, &tvel)

            arr = np.array([x,y,z,vx,vy,vz]).T
            arr[:,0] += f.x
            arr[:,1] += f.y
            arr[:,2] += f.z
            np.savetxt(snap_filename, arr)

            step_vel(config, b, -0.5*config.dt, &tnow, &tvel)

            j += 1

    # for i in range(1):
    #     print("xyz", x[i], y[i], z[i])
    #     print("vxyz", vx[i], vy[i], vz[i])
    #     print("axyz", ax[i], ay[i], az[i])
    #     print()
    # return


