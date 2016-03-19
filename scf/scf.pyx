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

from gary.potential.cpotential cimport _CPotential, valuefunc, gradientfunc

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
                 valuefunc vf, gradientfunc gf, double *parvec,
                 double extern_strength, double *tnow, int *firstc) nogil

    void frame(int iter, Config config, Bodies b, COMFrame *f) nogil

    void step_vel(Config config, Bodies b, double dt,
                  double *tnow, double *tvel) nogil

    void step_pos(Config config, Bodies b, double dt,
                  double *tnow, double *tvel) nogil

    void tidal_start(int iter, Config config, Bodies b, Placeholders p, COMFrame *f,
                     valuefunc vf, gradientfunc gf, double *parvec,
                     double *tnow, double *tpos, double *tvel) nogil

    void step_system(int iter, Config config, Bodies b, Placeholders p, COMFrame *f,
                     valuefunc vf, gradientfunc gf, double *parvec,
                     double *tnow, double *tpos, double *tvel) nogil

    void check_progenitor(int iter, Config config, Bodies b, Placeholders p, COMFrame *f,
                          valuefunc vf, gradientfunc gf, double *parvec, double *tnow) nogil

# needed for some reason
cdef extern from "src/helpers.h":
    void indexx(int n, double *arrin, int *indx) nogil

# ----------------------------------------------------------------------------

def write_snap(output_file, i, j, t, pos, vel, tub):
    # save snapshot to output file
    with h5py.File(output_file, 'r+') as out_f:
        g = out_f.create_group('/snapshots/{}'.format(j))
        g.attrs['t'] = t
        g.attrs['step'] = i
        g.create_dataset('pos', dtype=np.float64, shape=pos.shape, data=pos)
        g.create_dataset('vel', dtype=np.float64, shape=vel.shape, data=vel)
        g.create_dataset('tub', dtype=np.float64, shape=tub.shape, data=tub)

    logger.debug("\t...wrote snapshot {} to output file".format(j))

# TODO: WAT DO ABOUT POTENTIAL!?!
# TODO: is there some bug with the first snapshot output being offset in velocity?
def run_scf(_CPotential cpotential,
            w0, bodies, mass_scale, length_scale,
            dt, n_steps, t0, n_snapshot, n_recenter, n_tidal,
            nmax, lmax, zero_odd, zero_even, self_gravity,
            output_file):
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

        # frame position, velocity at all times
        double[:,::1] frame_xyz = np.zeros((3,n_steps+1))
        double[:,::1] frame_vxyz = np.zeros((3,n_steps+1))

        valuefunc vf = cpotential.c_value
        gradientfunc gf = cpotential.c_gradient
        double *parvec = &(cpotential._parameters[0])

    if os.path.exists(output_file):
        raise ValueError("Output file '{}' already exists.".format(output_file))

    # TODO: fucked up unit stuff
    _G = G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value
    X = (_G / ru**3 * mu)**-0.5
    time_unit = u.Unit("{:08f} Myr".format(X))
    if hasattr(dt, 'unit'):
        dt = dt.to(time_unit).value

    if hasattr(t0, 'unit'):
        t0 = t0.to(time_unit).value

    # store input parameters in the output file
    with h5py.File(output_file, 'w') as out_f:
        units = out_f.create_group('units')
        units.attrs['time'] = str(time_unit)
        units.attrs['length'] = str('{} kpc'.format(ru))
        units.attrs['mass'] = str('{} Msun'.format(mu))

        par = out_f.create_group('parameters')
        par.attrs['n_bodies'] = N

        par.attrs['n_steps'] = n_steps
        par.attrs['dt'] = dt

        par.attrs['n_recenter'] = n_recenter
        par.attrs['n_snapshot'] = n_snapshot
        par.attrs['n_tidal'] = n_tidal
        par.attrs['self_gravity'] = self_gravity
        par.attrs['zero_odd'] = zero_odd
        par.attrs['zero_even'] = zero_even
        par.attrs['nmax'] = nmax
        par.attrs['lmax'] = lmax

        snaps = out_f.create_group('snapshots')

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

    acc_pot(config, b, p, &f, vf, gf, parvec, extern_strength, &tnow, &firstc)
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
        tidal_start(i, config, b, p, &f,
                    vf, gf, parvec, &tnow, &tpos, &tvel)
        logger.debug("Tidal start: {}".format(i+1));

    # Synchronize the velocities with the positions
    step_vel(config, b, -0.5*config.dt, &tnow, &tvel)
    frame(0, config, b, &f)
    check_progenitor(0, config, b, p, &f, vf, gf, parvec, &tnow)

    # write initial positions out
    write_snap(output_file, i=0, j=0, t=tnow,
               pos=np.vstack((np.array(x)+f.x, np.array(y)+f.y, np.array(z)+f.z)),
               vel=np.vstack((np.array(vx), np.array(vy), np.array(vz))),
               tub=tub)

    # Reset the velocities to being 1/2 step ahead of the positions
    step_vel(config, b, 0.5*config.dt, &tnow, &tvel)
    #
    # ------------------------------------------------------------------------

    frame_xyz[0,0] = f.x
    frame_xyz[1,0] = f.y
    frame_xyz[2,0] = f.z
    frame_vxyz[0,0] = f.vx
    frame_vxyz[1,0] = f.vy
    frame_vxyz[2,0] = f.vz

    j = 1
    last_t = 0.
    for i in range(config.n_steps):
        PyErr_CheckSignals()
        step_system(i, config, b, p, &f, vf, gf, parvec, &tnow, &tpos, &tvel)
        logger.debug("Step: {}".format(i+1));

        frame_xyz[0,i+1] = f.x
        frame_xyz[1,i+1] = f.y
        frame_xyz[2,i+1] = f.z
        frame_vxyz[0,i+1] = f.vx
        frame_vxyz[1,i+1] = f.vy
        frame_vxyz[2,i+1] = f.vz

        if config.n_snapshot > 0 and (((i+1) % config.n_snapshot) == 0 and i > 0):
            step_vel(config, b, -0.5*config.dt, &tnow, &tvel)
            write_snap(output_file, i+1, j, t=tnow,
                       pos=np.vstack((np.array(x)+f.x, np.array(y)+f.y, np.array(z)+f.z)),
                       vel=np.vstack((np.array(vx), np.array(vy), np.array(vz))),
                       tub=tub)
            step_vel(config, b, 0.5*config.dt, &tnow, &tvel)
            j += 1
            last_t = tnow

    if (tnow - last_t) > 0.1*dt:
        step_vel(config, b, -0.5*config.dt, &tnow, &tvel)
        write_snap(output_file, i, j, t=tnow,
                   pos=np.vstack((np.array(x)+f.x, np.array(y)+f.y, np.array(z)+f.z)),
                   vel=np.vstack((np.array(vx), np.array(vy), np.array(vz))),
                   tub=tub)
        step_vel(config, b, 0.5*config.dt, &tnow, &tvel)

    with h5py.File(output_file, 'r+') as out_f:
        out_f.create_dataset('/cen/pos', dtype=np.float64, shape=np.array(frame_xyz).shape,
                             data=np.array(frame_xyz))
        out_f.create_dataset('/cen/vel', dtype=np.float64, shape=np.array(frame_vxyz).shape,
                             data=np.array(frame_vxyz))
