# coding: utf-8
# cython: debug=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

# Third-party
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

from libc.math cimport fabs, atan2, sqrt

from gala.potential.potential.cpotential cimport (CPotential,
                                                  c_potential,
                                                  c_gradient)

from structs cimport Config, Placeholders, Bodies, COMFrame, step_pos, step_vel
from .acceleration cimport update_acceleration

# -----------------------------------------------------------------------------

cdef void recenter_frame(int iter, Config config, Bodies b, COMFrame *f):
    """
    Recompute the center-of-mass frame.

    Parameters
    ----------
    iter : int
        The current iteration.
    config : Config (struct)
        Struct containing configuration parameters.
    b : Bodies (struct)
        Struct of pointers to arrays that contain information about the mass
        particles (the bodies).
    f : COMFrame (struct)
        Pointer to center-of-pass struct containing the phase-space position
        of the frame.
    """

    cdef:
        # TODO: make the minimum number of bound particles a configurable param
        # Use only the most bound particles to determine the new frame
        int nend = config.n_bodies / 10
        int i,j,k

        double total_mass = 0.
        double center_of_mass[6] # x,y,z,vx,vy,vz

    # Initialization
    if (nend < 1024):
        nend = 1024 # Always use at least 1024 particles

    for j in range(6):
        center_of_mass[j] = 0.

    for i in range(nend):
        k = f.pot_idx[i]
        center_of_mass[0] = center_of_mass[0] + b.mass[k]*b.x[k]
        center_of_mass[1] = center_of_mass[1] + b.mass[k]*b.y[k]
        center_of_mass[2] = center_of_mass[2] + b.mass[k]*b.z[k]

        center_of_mass[3] = center_of_mass[3] + b.mass[k]*b.vx[k]
        center_of_mass[4] = center_of_mass[4] + b.mass[k]*b.vy[k]
        center_of_mass[5] = center_of_mass[5] + b.mass[k]*b.vz[k]

        total_mass = total_mass + b.mass[k]

    # Divide out total mass
    for j in range(6):
        center_of_mass[j] = center_of_mass[j] / total_mass

    # Update frame and shift to center on the minimum of the potential
    (f.x) = f.x + center_of_mass[0]
    (f.y) = f.y + center_of_mass[1]
    (f.z) = f.z + center_of_mass[2]
    (f.vx) = f.vx + center_of_mass[3]
    (f.vy) = f.vy + center_of_mass[4]
    (f.vz) = f.vz + center_of_mass[5]

    # Shift all positions to be oriented on the center-of-mass frame
    for k in range(config.n_bodies):
        b.x[k] = b.x[k] - center_of_mass[0]
        b.y[k] = b.y[k] - center_of_mass[1]
        b.z[k] = b.z[k] - center_of_mass[2]

        b.vx[k] = b.vx[k] - center_of_mass[3]
        b.vy[k] = b.vy[k] - center_of_mass[4]
        b.vz[k] = b.vz[k] - center_of_mass[5]

cdef void check_progenitor(int iter, Config config, Bodies b, Placeholders p,
                      COMFrame *f, CPotential *pot, double *tnow):
    """
    Iteratively determine which bodies are still bound to the progenitor
    and determine whether it is still self-gravitating.

    Roughly equivalent to 'findrem' in Fortran.

    Parameters
    ----------
    iter : int
        The current iteration.
    config : Config (struct)
        Struct containing configuration parameters.
    b : Bodies (struct)
        Struct of pointers to arrays that contain information about the mass
        particles (the bodies).
    p : Placeholders (struct)
        Struct of pointers to placeholder arrays used in the BFE calculations.
    f : COMFrame (struct)
        Pointer to center-of-pass struct containing the phase-space position
        of the frame.
    tnow : double
        The current simulation time.
    """

    cdef:
        double m_prog = 10000000000.;
        double m_safe;
        double vx_rel, vy_rel, vz_rel;
        int k,n;
        int N_MASS_ITER = 30; # TODO: set in config?
        int not_firstc = 0;
        int broke = 0;

    for k in range(config.n_bodies):
        vx_rel = b.vx[k] - f.vx
        vy_rel = b.vy[k] - f.vy
        vz_rel = b.vz[k] - f.vz

        p.kin0[k] = 0.5*(vx_rel*vx_rel + vy_rel*vy_rel + vz_rel*vz_rel); # relative
        p.pot0[k] = b.Epot_bfe[k] # potential energy in satellite
        p.ax0[k] = b.ax[k]
        p.ay0[k] = b.ay[k]
        p.az0[k] = b.az[k]

    # iteratively shave off unbound particles to find prog. mass
    for n in range(N_MASS_ITER):
        m_safe = m_prog
        m_prog = 0.

        for k in range(config.n_bodies):
            if (p.kin0[k] > fabs(b.Epot_bfe[k])):
                # relative kinetic energy > potential
                b.ibound[k] = 0
                if b.tub[k] == 0:
                    b.tub[k] = tnow[0]

            else:
                m_prog = m_prog + b.mass[k]

        if m_safe <= m_prog:
            broke = 1
            break

        # Find new accelerations with unbound stuff removed
        update_acceleration(config, b, p, f,
                            pot, 1., tnow, &not_firstc)

    # if the loop above didn't break, progenitor is dissolved?
    if broke == 0:
        m_prog = 0.

    for k in range(config.n_bodies):
        b.Epot_bfe[k] = p.pot0[k]
        b.ax[k] = p.ax0[k]
        b.ay[k] = p.ay0[k]
        b.az[k] = p.az0[k]

    if m_prog == 0:
        config.selfgravitating = 0
    f.m_prog = m_prog

cdef void tidal_start(int iter, Config config, Bodies b, Placeholders p,
                      COMFrame *f, CPotential *pot, double *tnow, double *tpos,
                      double *tvel):
    """
    Slowly ramp up the tidal field.
    """
    cdef:
        double v_cm[3]
        double a_cm[3]
        double mtot, t_tidal, strength
        int i,k
        int not_firstc = 0

    # Advance position by one step
    step_pos(config, b, config.dt, tnow, tpos)

    # Find center-of-mass vel. and acc.
    for i in range(3):
        v_cm[i] = 0.
        a_cm[i] = 0.
    mtot = 0.

    for k in range(config.n_bodies):
        v_cm[0] = v_cm[0] + b.mass[k]*b.vx[k]
        v_cm[1] = v_cm[1] + b.mass[k]*b.vy[k]
        v_cm[2] = v_cm[2] + b.mass[k]*b.vz[k]

        a_cm[0] = a_cm[0] + b.mass[k]*b.ax[k]
        a_cm[1] = a_cm[1] + b.mass[k]*b.ay[k]
        a_cm[2] = a_cm[2] + b.mass[k]*b.az[k]

        mtot = mtot + b.mass[k]

    for i in range(3):
        v_cm[i] = v_cm[i] / mtot
        a_cm[i] = a_cm[i] / mtot

    # Retard position and velocity by one step relative to center of mass??
    for k in range(config.n_bodies):
        b.x[k] = b.x[k] - v_cm[0]*config.dt
        b.y[k] = b.y[k] - v_cm[1]*config.dt
        b.z[k] = b.z[k] - v_cm[2]*config.dt

        b.vx[k] = b.vx[k] - a_cm[0]*config.dt
        b.vy[k] = b.vy[k] - a_cm[1]*config.dt
        b.vz[k] = b.vz[k] - a_cm[2]*config.dt

    # Increase tidal field
    t_tidal = (<double>iter + 1.) / (<double>config.n_tidal)
    strength = (-2.*t_tidal + 3.)*t_tidal*t_tidal

    # Find new accelerations
    update_acceleration(config, b, p, f, pot,
                        strength, tnow, &not_firstc)

    # Advance velocity by one step
    step_vel(config, b, config.dt, tnow, tvel)

    # Reset times
    tvel[0] = 0.5*config.dt
    tpos[0] = 0.
    tnow[0] = 0.
