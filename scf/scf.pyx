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
from astropy import log as logger
import astropy.units as u
from astropy.constants import G, M_sun
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

    void tidal_start(int iter, Config config, Bodies b, Placeholders p, COMFrame *f,
                     double *tnow, double *tpos, double *tvel) nogil

    void step_system(int iter, Config config, Bodies b, Placeholders p, COMFrame *f,
                     double *tnow, double *tpos, double *tvel) nogil

    void check_progenitor(int iter, Config config, Bodies b, Placeholders p,
                          COMFrame *f, double *tnow) nogil

# needed for some reason
cdef extern from "src/helpers.h":
    void indexx(int n, double *arrin, int *indx) nogil

# TODO: WAT DO ABOUT POTENTIAL!?!
def run_scf(w0, bodies, mass_scale, length_scale,
            dt, n_steps, t0, n_snapshot, n_recenter, n_tidal,
            nmax, lmax, zero_odd, zero_even, self_gravity,
            output_path):
    cdef:
        int firstc = 1
        Config config
        Placeholders p
        Bodies b
        COMFrame f

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
        # TODO: figure out a more sensible way to do this
        double ru = length_scale.to(u.kpc).value
        double mu = mass_scale.to(u.Msun).value
        double gee = G.decompose([u.cm,u.g,u.s]).value
        double msun = M_sun.to(u.g).value
        double cmpkpc = (1*u.kpc).to(u.cm).value
        double secperyr = (1*u.year).to(u.second).value
        double tu = sqrt((cmpkpc*ru)**3 / (msun*mu*gee))
        double vu = (ru*cmpkpc*1.e-5)/tu

        # for turning on the potential:
        double extern_strength

        # timing
        double tnow, tpos, tvel

    # TODO: fucked up unit stuff
    _G = 1.
    X = np.sqrt(ru**3 * mu / _G)
    time_unit = u.Unit("{:08f} Myr".format(X))
    if hasattr(dt, 'unit'):
        dt = dt.to(time_unit).value

    # Configuration stuff
    config.n_bodies = N
    config.n_steps = int(n_steps)
    config.dt = float(dt)
    config.t0 = float(t0)
    config.n_recenter = int(n_recenter)
    config.n_snapshot = int(n_snapshot)
    config.n_tidal = int(n_tidal)
    config.selfgravitating = int(self_gravity)
    config.nmax = int(nmax)
    config.lmax = int(lmax)
    config.zeroodd = int(zero_odd)
    config.zeroeven = int(zero_even)
    config.G = 1. # HACK: do we really need to let the user set this?
    config.ru = ru
    config.mu = mu
    config.tu = tu
    config.vu = vu

    # the position and velocity of the progenitor
    xyz = w0.pos.to(u.kpc).value
    vxyz = w0.vel.to(u.km/u.s).value
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

    # ------------------------------------------------------------------------
    # this stuff follows `initsys`
    #
    tnow = config.t0
    tpos = tnow
    tvel = tnow

    # frame in simulation units
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
    #
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Tidal start: slowly turn on tidal field
    #
    for i in range(config.n_tidal):
        PyErr_CheckSignals()
        tidal_start(i, config, b, p, &f, &tnow, &tpos, &tvel)
        logger.debug("Tidal start: {}".format(i+1));

    # Synchronize the velocities with the positions
    step_vel(config, b, -0.5*config.dt, &tnow, &tvel)
    frame(0, config, b, &f)
    check_progenitor(0, config, b, p, &f, &tnow)

    # Reset the velocities to being 1/2 step ahead of the positions
    step_vel(config, b, -0.5*config.dt, &tnow, &tvel)
    #
    # ------------------------------------------------------------------------

    # TODO: if config.n_snapshot is 0, only output final state
    j = 0
    for i in range(config.n_steps):
        PyErr_CheckSignals()
        step_system(i, config, b, p, &f, &tnow, &tpos, &tvel)

        if config.n_snapshot > 0 and (((i+1) % config.n_snapshot) == 0 or i == 0):
            step_vel(config, b, 0.5*config.dt, &tnow, &tvel)

            # TODO: save snapshots
            # arr = np.array([x,y,z,vx,vy,vz]).T
            # arr[:,0] += f.x
            # arr[:,1] += f.y
            # arr[:,2] += f.z
            # np.savetxt(snap_filename, arr)

            step_vel(config, b, -0.5*config.dt, &tnow, &tvel)

            j += 1

        logger.debug("Step: {}".format(i+1));

    return np.vstack((np.array(x), np.array(y), np.array(z)))
