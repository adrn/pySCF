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

from gala.potential.potential.cpotential cimport CPotentialWrapper

cdef extern from "math.h":
    double sqrt(double)

cdef extern from "potential/src/cpotential.h":
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
        double *Epot_ext;
        double *Epot_bfe;
        double *Ekin;
        double *mass
        int *ibound
        double *tub

    ctypedef struct COMFrame:
        double m_prog
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
        double[::1] x = np.ascontiguousarray(bodies.xyz.value[0][:])
        double[::1] y = np.ascontiguousarray(bodies.xyz.value[1][:])
        double[::1] z = np.ascontiguousarray(bodies.xyz.value[2][:])
        double[::1] vx = np.ascontiguousarray(bodies.v_xyz.value[0][:])
        double[::1] vy = np.ascontiguousarray(bodies.v_xyz.value[1][:])
        double[::1] vz = np.ascontiguousarray(bodies.v_xyz.value[2][:])
        double[::1] mass = np.ones(N) / N
        int[::1] ibound = np.ones(N, dtype=np.int32)
        double[::1] tub = np.zeros(N)
        double[::1] ax = np.zeros(N)
        double[::1] ay = np.zeros(N)
        double[::1] az = np.zeros(N)
        double[::1] Epot_bfe = np.zeros(N) # (internal) potential energy
        double[::1] Epot_ext = np.zeros(N) # (external) potential energy
        double[::1] Ekin = np.zeros(N) # kinetic energy

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
    config.G = 1.

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
    b.Epot_bfe = &Epot_bfe[0]
    b.Epot_ext = &Epot_ext[0]
    b.Ekin = &Ekin[0]
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
        'ax': ax,
        'ay': ay,
        'az': az,
        'pot_ext': Epot_ext,
        'pot_bfe': Epot_bfe
    }

cpdef _test_tidal_start(CPotentialWrapper cp, w0, bodies, n_tidal, length_scale, mass_scale):
    cdef:
        int firstc = 1
        Config config
        Placeholders p
        Bodies b
        COMFrame f

        int nmax = 6
        int lmax = 4

        # Read in the phase-space positions of the N bodies
        int N = bodies.shape[0]
        double[::1] x = np.ascontiguousarray(bodies.xyz.value[0][:])
        double[::1] y = np.ascontiguousarray(bodies.xyz.value[1][:])
        double[::1] z = np.ascontiguousarray(bodies.xyz.value[2][:])
        double[::1] vx = np.ascontiguousarray(bodies.v_xyz.value[0][:])
        double[::1] vy = np.ascontiguousarray(bodies.v_xyz.value[1][:])
        double[::1] vz = np.ascontiguousarray(bodies.v_xyz.value[2][:])
        double[::1] mass = np.ones(N) / N
        int[::1] ibound = np.ones(N, dtype=np.int32)
        double[::1] tub = np.zeros(N)
        double[::1] ax = np.zeros(N)
        double[::1] ay = np.zeros(N)
        double[::1] az = np.zeros(N)
        double[::1] Epot_bfe = np.zeros(N) # (internal) potential energy
        double[::1] Epot_ext = np.zeros(N) # (external) potential energy
        double[::1] Ekin = np.zeros(N) # kinetic energy

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
        double[::1] pot0 = np.zeros(N) # placeholder (internal) potential energy
        double[::1] kin0 = np.zeros(N) # placeholder kinetic energy
        double[::1] ax0 = np.zeros(N) # placeholder
        double[::1] ay0 = np.zeros(N) # placeholder
        double[::1] az0 = np.zeros(N) # placeholder

        int i,j
        int[::1] tmp = np.zeros(N, dtype=np.int32) # temporary array for indexing

        # index array for sorting particles on potential value
        int[::1] pot_idx = np.zeros(N, dtype=np.int32)

        # sim units
        # TODO: figure out a more sensible way to do this
        double ru = length_scale.to(u.kpc).value
        double mu = mass_scale.to(u.Msun).value
        # HACK: for now, set to fortran values
        # double gee = G.decompose([u.cm,u.g,u.s]).value
        # double msun = M_sun.to(u.g).value
        # double cmpkpc = (1*u.kpc).to(u.cm).value
        # double secperyr = (1*u.year).to(u.second).value
        double gee = 6.67384e-8
        double msun = 1.9891e33
        double cmpkpc = 3.0856776e21
        double secperyr = 3.15576e7

        double tu = sqrt((cmpkpc*ru)**3 / (msun*mu*gee))
        double vu = (ru*cmpkpc*1.e-5)/tu

        # for turning on the potential:
        double extern_strength = 0.

        # timing
        double tnow, tpos, tvel

        # hard set (in simulation units)
        double dt = 0.25
        double t0 = 0.

    # Configuration stuff
    config.n_bodies = N
    config.n_steps = 0
    config.dt = float(dt)
    config.t0 = float(t0)
    config.n_recenter = 1
    config.n_snapshot = 1
    config.n_tidal = int(n_tidal)
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
    xyz = w0.xyz.to(u.kpc).value
    vxyz = w0.v_xyz.to(u.km/u.s).value
    f.x = float(xyz[0])
    f.y = float(xyz[1])
    f.z = float(xyz[2])
    f.vx = float(vxyz[0])
    f.vy = float(vxyz[1])
    f.vz = float(vxyz[2])
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
    b.Epot_bfe = &Epot_bfe[0]
    b.Epot_ext = &Epot_ext[0]
    b.Ekin = &Ekin[0]
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
    p.pot0 = &pot0[0]
    p.kin0 = &kin0[0]
    p.ax0 = &ax0[0]
    p.ay0 = &ay0[0]
    p.az0 = &az0[0]

    # ------------------------------------------------------------------------
    # this stuff follows `initsys`
    #
    tnow = config.t0
    tpos = tnow
    tvel = tnow

    # frame in simulation units
    f.m_prog = 1.
    f.x = f.x / ru
    f.y = f.y / ru
    f.z = f.z / ru
    f.vx = f.vx / vu
    f.vy = f.vy / vu
    f.vz = f.vz / vu

    for i in range(N):
        Ekin[i] = 0.5 * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
        vx[i] = vx[i] + f.vx
        vy[i] = vy[i] + f.vy
        vz[i] = vz[i] + f.vz

    acc_pot(config, b, p, &f, &(cp.cpotential), extern_strength, &tnow, &firstc)

    # sort particles index array on potential value
    tmp = np.argsort(Epot_bfe).astype(np.int32)
    for i in range(N):
        pot_idx[i] = tmp[i]
    frame(0, config, b, &f)

    # initialize velocities (take a half step in time)
    step_vel(config, b, 0.5*config.dt, &tnow, &tvel)
    v_init = np.vstack((np.array(vx, copy=True), np.array(vy, copy=True), np.array(vz, copy=True)))
    #
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Tidal start: slowly turn on tidal field
    #
    for i in range(config.n_tidal):
        print("tidal start {}".format(i+1))
        PyErr_CheckSignals()
        tidal_start(i, config, b, p, &f,
                    &(cp.cpotential), &tnow, &tpos, &tvel)

        if i == 0:
            xyz_one_step = np.vstack((np.array(x, copy=True), np.array(y, copy=True), np.array(z, copy=True)))
            vxyz_one_step = np.vstack((np.array(vx, copy=True), np.array(vy, copy=True), np.array(vz, copy=True)))

    # Synchronize the velocities with the positions
    step_vel(config, b, -0.5*config.dt, &tnow, &tvel)

    # save pos, vel at end
    xyz_end = np.vstack((np.array(x, copy=True), np.array(y, copy=True), np.array(z, copy=True)))
    vxyz_end = np.vstack((np.array(vx, copy=True), np.array(vy, copy=True), np.array(vz, copy=True)))

    # sort particles index array on potential value
    tmp = np.argsort(Epot_bfe).astype(np.int32)
    for i in range(N):
        pot_idx[i] = tmp[i]

    frame(0, config, b, &f)
    check_progenitor(0, config, b, p, &f, &(cp.cpotential), &tnow)

    return {
        'v_after_init': v_init,
        'xyz_one_step': xyz_one_step,
        'vxyz_one_step': vxyz_one_step,
        'xyz_end': xyz_end,
        'vxyz_end': vxyz_end,
        'm_prog': f.m_prog,
        'frame_xyz': [f.x, f.y, f.z],
        'frame_vxyz': [f.vx, f.vy, f.vz],
    }
