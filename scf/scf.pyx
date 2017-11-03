# coding: utf-8
# cython: debug=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

# Standard library
import os
import sys

# Third-party
import astropy.units as u
from astropy.constants import G
import h5py
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython
from cpython.exc cimport PyErr_CheckSignals
from libc.math cimport sqrt

from gala.potential.potential.cpotential import CPotentialBase
from gala.potential.potential.cpotential cimport (CPotentialWrapper,
                                                  CPotential)

from .log import logger
from .acceleration cimport update_acceleration
from .progenitor cimport recenter_frame, check_progenitor, tidal_start
from structs cimport Config, Placeholders, Bodies, COMFrame, step_pos, step_vel

# needed for some reason
cdef extern from "src/helpers.h":
    void indexx(int n, double *arrin, int *indx) nogil

# ----------------------------------------------------------------------------

def write_snap(output_file, i, j, t, xyz, vxyz, tub,
               Ekin, Epot_bfe, Epot_ext,
               write_energy=False, output_dtype=np.float64):

    """
    Write a simulation snapshop.

    Parameters
    ----------
    xyz : tuple
    vxyz : tuple
    """

    xyz = np.vstack((np.array(xyz[0]),
                     np.array(xyz[1]),
                     np.array(xyz[2]))).astype(output_dtype)
    vxyz = np.vstack((np.array(vxyz[0]),
                      np.array(vxyz[1]),
                      np.array(vxyz[2]))).astype(output_dtype)

    # save snapshot to output file
    with h5py.File(output_file, 'r+') as out_f:
        g = out_f.create_group('/snapshots/{0}'.format(j))
        g.attrs['t'] = t
        g.attrs['step'] = i
        g.create_dataset('pos', data=xyz)
        g.create_dataset('vel', data=vxyz)
        g.create_dataset('tub', data=np.array(tub).astype(output_dtype))

        if write_energy:
            eg = g.create_group('energy')
            eg.create_dataset('kinetic_int',
                              data=np.array(Ekin).astype(output_dtype))
            eg.create_dataset('potential_int',
                              data=np.array(Epot_bfe).astype(output_dtype))
            eg.create_dataset('potential_ext',
                              data=np.array(Epot_ext).astype(output_dtype))

    logger.debug("Writing snapshot {0}".format(j))

cdef void step_system(int iter, Config config, Bodies b, Placeholders p,
                      COMFrame *f, CPotential *pot, double *tnow, double *tpos,
                      double *tvel, int n_recenter):
    cdef:
        int not_firstc = 0
        double strength = 1.
        int[::1] tmp = np.zeros(config.n_bodies, dtype=np.int32)
        double[::1] wtf = np.zeros(config.n_bodies, dtype=np.float64)

    step_pos(config, b, config.dt, tnow, tpos)

    # sort particles index array on internal potential value
    for i in range(config.n_bodies):
        wtf[i] = b.Epot_bfe[i]

    tmp = np.argsort(wtf).astype(np.int32)
    for i in range(config.n_bodies):
        f.pot_idx[i] = tmp[i]

    if iter % n_recenter == 0:
        logger.debug("Recentering progenitor frame")
        recenter_frame(config, b, f)

    update_acceleration(config, b, p, f, pot,
                        strength, tnow, &not_firstc)
    step_vel(config, b, 0.5*config.dt, tnow, tvel)

def run_scf(CPotentialWrapper cp,
            w0, bodies, mass_scale, length_scale,
            dt, n_steps, t0, n_snapshot, n_recenter, n_tidal,
            nmax, lmax, zero_odd, zero_even, self_gravity, output_file,
            write_energy, show_progress=False, output_dtype=np.float64):
    cdef:
        int firstc = 1
        Config config
        Placeholders p
        Bodies b
        COMFrame f

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
        double[::1] pot0 = np.zeros(N) # placeholder (internal) potential energy
        double[::1] kin0 = np.zeros(N) # placeholder (internal) kinetic energy
        double[::1] ax0 = np.zeros(N) # placeholder
        double[::1] ay0 = np.zeros(N) # placeholder
        double[::1] az0 = np.zeros(N) # placeholder

        int i,j
        int[::1] tmp = np.zeros(N, dtype=np.int32) # temp array for indexing

        # index array for sorting particles on potential value
        int[::1] pot_idx = np.zeros(N, dtype=np.int32)

        # for turning on the potential:
        double extern_strength

        # timing
        double tnow, tpos, tvel

        # center-of-mass frame position, velocity at all timesteps
        double[:,::1] frame_xyz = np.zeros((3, n_steps+1))
        double[:,::1] frame_vxyz = np.zeros((3, n_steps+1))

    # Simulation units
    ru = length_scale.to(u.kpc)
    mu = mass_scale.to(u.Msun)
    tu = np.sqrt((ru**3) / (G*mu)).to(u.yr)
    vu = (ru/tu).to(u.km/u.s)

    if os.path.exists(output_file):
        raise ValueError("Output file '{0}' already exists."
                         .format(output_file))

    if hasattr(dt, 'unit'):
        dt = dt.to(tu).value

    if hasattr(t0, 'unit'):
        t0 = t0.to(tu).value

    # Store input parameters in the output file
    with h5py.File(output_file, 'w') as out_f:
        units = out_f.create_group('units')
        units.attrs['time'] = "{:.18f}".format(tu)
        units.attrs['length'] = "{:.18f}".format(ru)
        units.attrs['mass'] = "{:.18f}".format(mu)

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
    config.G = 1. # work in units where G = 1
    config.ru = ru.value
    config.mu = mu.value
    config.tu = tu.value
    config.vu = vu.value

    # The position and velocity of the progenitor
    xyz = w0.xyz.to(u.kpc).value
    vxyz = w0.v_xyz.to(u.km/u.s).value
    f.x = float(xyz[0])
    f.y = float(xyz[1])
    f.z = float(xyz[2])
    f.vx = float(vxyz[0])
    f.vy = float(vxyz[1])
    f.vz = float(vxyz[2])
    f.pot_idx = &pot_idx[0]

    # The N bodies: pointers to arrays containing initial conditions, etc.
    # Note that the positions (x,y,z) and velocities (vx,vy,vz) are *absolute*
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

    # Initial strength of tidal field
    if config.n_tidal > 0:
        # If we're going to slowly turn on the tidal field, start from 0
        extern_strength = 0.
    else:
        # If we don't ramp up the tidal field, start at full strength
        extern_strength = 1.

    # ------------------------------------------------------------------------
    # This stuff follows `initsys` in the original FORTRAN implementation
    # At this point, the particle positions and velocities are relative and
    # don't include the motion of the satellite itself.

    tnow = config.t0
    tpos = tnow
    tvel = tnow

    # Frame in simulation units
    f.m_prog = 1. # mass unit is the total mass of the progenitor
    f.x = f.x / config.ru
    f.y = f.y / config.ru
    f.z = f.z / config.ru
    f.vx = f.vx / config.vu
    f.vy = f.vy / config.vu
    f.vz = f.vz / config.vu

    frame_xyz[0,0] = f.x
    frame_xyz[1,0] = f.y
    frame_xyz[2,0] = f.z
    frame_vxyz[0,0] = f.vx
    frame_vxyz[1,0] = f.vy
    frame_vxyz[2,0] = f.vz

    # Compute initial (internal) kinetic energy of particles, add progenitor
    # velocity to particle velocities. Note the velocities at this point are
    # relative to the satellite frame - we add the frame velocity and position
    # now:
    for i in range(N):
        Ekin[i] = 0.5 * (vx[i]**2 + vy[i]**2 + vz[i]**2)

        b.x[i] = b.x[i] + f.x
        b.y[i] = b.y[i] + f.y
        b.z[i] = b.z[i] + f.z

        b.vx[i] = b.vx[i] + f.vx
        b.vy[i] = b.vy[i] + f.vy
        b.vz[i] = b.vz[i] + f.vz

    # Initialize accleration arrats (but external field might be zero if we are
    # doing tidal start)
    update_acceleration(config, b, p, &f, &(cp.cpotential), extern_strength,
                        &tnow, &firstc)

    # Sort particles index array on potential value (most bound particles first)
    # then do initial re-determination of center-of-mass frame. The sorting is
    # important for doing the frame determination!
    tmp = np.argsort(Epot_bfe).astype(np.int32)
    for i in range(N):
        pot_idx[i] = tmp[i]
    recenter_frame(config, b, &f)

    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Tidal start: slowly turn on tidal field

    # Initialize velocities (take a half step in time)
    step_vel(config, b, 0.5*config.dt, &tnow, &tvel)

    for i in range(config.n_tidal):
        PyErr_CheckSignals() # check for keyboard interrupts
        tidal_start(i, config, b, p, &f,
                    &(cp.cpotential), &tnow, &tpos, &tvel)
        logger.debug("Tidal start: {}".format(i+1))

    # Synchronize the velocities with the positions
    step_vel(config, b, -0.5*config.dt, &tnow, &tvel)

    # Sort particles index array on potential value
    tmp = np.argsort(Epot_bfe).astype(np.int32)
    for i in range(N):
        pot_idx[i] = tmp[i] # (f->pot_idx) points to this
    recenter_frame(config, b, &f)
    check_progenitor(0, config, b, p, &f, &(cp.cpotential), &tnow)

    # Write initial positions out after tidal start
    write_snap(output_file, i=0, j=0, t=tnow, tub=tub,
               xyz=(x, y, z), vxyz=(vx, vy, vz),
               Ekin=Ekin, Epot_bfe=Epot_bfe, Epot_ext=Epot_ext,
               write_energy=write_energy, output_dtype=output_dtype)

    # Reset the velocities to being 1/2 step ahead of the positions
    step_vel(config, b, 0.5*config.dt, &tnow, &tvel)

    # ------------------------------------------------------------------------

    j = 1 # snapshot number (we already wrote 0)
    wrote = None

    if show_progress:
        from tqdm import trange
        iter_func = trange(config.n_steps)
    else:
        iter_func = range(config.n_steps)

    for i in iter_func:
        PyErr_CheckSignals()

        # Steps position by 1 step, velocity by 1/2 step
        step_system(i+1, config, b, p, &f, &(cp.cpotential), &tnow, &tpos,
                    &tvel, config.n_recenter)

        frame_xyz[0,i+1] = f.x
        frame_xyz[1,i+1] = f.y
        frame_xyz[2,i+1] = f.z
        frame_vxyz[0,i+1] = f.vx
        frame_vxyz[1,i+1] = f.vy
        frame_vxyz[2,i+1] = f.vz

        check_progenitor(i, config, b, p, &f, &(cp.cpotential), &tnow)
        logger.debug("Step {0}, bound mass: {1:.0%}".format(i, f.m_prog))

        if config.n_snapshot > 0 and (((i+1) % config.n_snapshot) == 0 and i>0):
            write_snap(output_file, i+1, j, t=tnow, tub=tub,
                       xyz=(x, y, z), vxyz=(vx, vy, vz),
                       Ekin=Ekin, Epot_bfe=Epot_bfe, Epot_ext=Epot_ext,
                       write_energy=write_energy, output_dtype=output_dtype)
            j += 1
            wrote = True

        else:
            wrote = False

        # Complete the full step in velocity
        step_vel(config, b, 0.5*config.dt, &tnow, &tvel)

    # Always write the last timestep (if it wasn't written already), even if
    # n_snapshot==0
    if not wrote:
        write_snap(output_file, i+1, j, t=tnow, tub=tub,
                   xyz=(x, y, z), vxyz=(vx, vy, vz),
                   Ekin=Ekin, Epot_bfe=Epot_bfe, Epot_ext=Epot_ext,
                   write_energy=write_energy, output_dtype=output_dtype)

    with h5py.File(output_file, 'r+') as out_f:
        out_f.create_dataset('/cen/pos',
                             data=np.array(frame_xyz).astype(output_dtype))
        out_f.create_dataset('/cen/vel',
                             data=np.array(frame_vxyz).astype(output_dtype))
