#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <math.h>
#include <Python.h>
#include "potential/src/cpotential.h" // from gala/potential
#include "helpers.h"
#include "scf.h"

void recenter_frame(int iter, Config config, Bodies b, COMFrame *f) {
    /*
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
    */

    // TODO: make the minimum number of bound particles a configurable param
    // Use only the most bound particles to determine the new frame
    int nend = config.n_bodies / 10;
    if (nend < 1024) nend = 1024; // Always use at least 1024 particles
    int i,j,k;

    double total_mass = 0.;
    double center_of_mass[6]; // x,y,z,vx,vy,vz
    for (j=0; j<6; j++) {
        center_of_mass[j] = 0.;
    }

    for (i=0; i<nend; i++) {
        k = (f->pot_idx)[i];
        center_of_mass[0] = center_of_mass[0] + b.mass[k]*b.x[k];
        center_of_mass[1] = center_of_mass[1] + b.mass[k]*b.y[k];
        center_of_mass[2] = center_of_mass[2] + b.mass[k]*b.z[k];

        center_of_mass[3] = center_of_mass[3] + b.mass[k]*b.vx[k];
        center_of_mass[4] = center_of_mass[4] + b.mass[k]*b.vy[k];
        center_of_mass[5] = center_of_mass[5] + b.mass[k]*b.vz[k];

        total_mass = total_mass + b.mass[k];
    }

    // Divide out total mass
    for (j=0; j<6; j++) {
        center_of_mass[j] = center_of_mass[j] / total_mass;
    }

    // Update frame and shift to center on the minimum of the potential
    (f->x) = (f->x) + center_of_mass[0];
    (f->y) = (f->y) + center_of_mass[1];
    (f->z) = (f->z) + center_of_mass[2];
    (f->vx) = (f->vx) + center_of_mass[3];
    (f->vy) = (f->vy) + center_of_mass[4];
    (f->vz) = (f->vz) + center_of_mass[5];

    // Shift all positions to be oriented on the center-of-mass frame
    for (k=0; k<config.n_bodies; k++) {
        b.x[k] = b.x[k] - center_of_mass[0];
        b.y[k] = b.y[k] - center_of_mass[1];
        b.z[k] = b.z[k] - center_of_mass[2];

        b.vx[k] = b.vx[k] - center_of_mass[3];
        b.vy[k] = b.vy[k] - center_of_mass[4];
        b.vz[k] = b.vz[k] - center_of_mass[5];
    }
}

void check_progenitor(int iter, Config config, Bodies b, Placeholders p, COMFrame *f,
                      CPotential *pot, double *tnow) {
    /*
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
    */
    double m_prog, m_safe;
    double vx_rel, vy_rel, vz_rel;
    int k,n;
    int N_MASS_ITER = 30; // TODO: set in config?
    int not_firstc = 0;
    int broke = 0;

    m_prog = 10000000000.;

    for (k=0; k<config.n_bodies; k++) {
        vx_rel = b.vx[k] - (f->vx);
        vy_rel = b.vy[k] - (f->vy);
        vz_rel = b.vz[k] - (f->vz);

        p.kin0[k] = 0.5*(vx_rel*vx_rel + vy_rel*vy_rel + vz_rel*vz_rel); // relative
        p.pot0[k] = b.Epot_bfe[k]; // potential energy in satellite
        p.ax0[k] = b.ax[k];
        p.ay0[k] = b.ay[k];
        p.az0[k] = b.az[k];
    }

    // iteratively shave off unbound particles to find prog. mass
    for (n=0; n<N_MASS_ITER; n++) {
        m_safe = m_prog;
        m_prog = 0.;

        for (k=0; k<config.n_bodies; k++) {
            if (p.kin0[k] > fabs(b.Epot_bfe[k])) { // relative kinetic energy > potential
                b.ibound[k] = 0;
                if (b.tub[k] == 0) b.tub[k] = *tnow;
            } else {
                m_prog = m_prog + b.mass[k];
            }
        }

        if (m_safe <= m_prog) {
            broke = 1;
            break;
        }

        // Find new accelerations with unbound stuff removed
        update_acceleration(config, b, p, f,
                            pot, 1., tnow, &not_firstc);
    }

    // if the loop above didn't break, progenitor is dissolved?
    if (broke == 0) m_prog = 0.;

    for (k=0; k<config.n_bodies; k++) {
        b.Epot_bfe[k] = p.pot0[k];
        b.ax[k] = p.ax0[k];
        b.ay[k] = p.ay0[k];
        b.az[k] = p.az0[k];
    }

    if (m_prog == 0) config.selfgravitating = 0;
    (f->m_prog) = m_prog;

}

void tidal_start(int iter, Config config, Bodies b, Placeholders p, COMFrame *f,
                 CPotential *pot, double *tnow, double *tpos, double *tvel) {
    double v_cm[3], a_cm[3], mtot, t_tidal, strength;
    int i,k;
    int not_firstc = 0;

    // printf("v100 %d %.14f %.14f %.14f\n", iter, b.vx[100], b.vy[100], b.vz[100]);

    // Advance position by one step
    step_pos(config, b, config.dt, tnow, tpos);

    // Find center-of-mass vel. and acc.
    for (i=0; i<3; i++) {
        v_cm[i] = 0.;
        a_cm[i] = 0.;
    }
    mtot = 0.;

    for (k=0; k<config.n_bodies; k++) {
        v_cm[0] = v_cm[0] + b.mass[k]*b.vx[k];
        v_cm[1] = v_cm[1] + b.mass[k]*b.vy[k];
        v_cm[2] = v_cm[2] + b.mass[k]*b.vz[k];

        a_cm[0] = a_cm[0] + b.mass[k]*b.ax[k];
        a_cm[1] = a_cm[1] + b.mass[k]*b.ay[k];
        a_cm[2] = a_cm[2] + b.mass[k]*b.az[k];

        mtot = mtot + b.mass[k];
    }

    for (i=0; i<3; i++) {
        v_cm[i] = v_cm[i]/mtot;
        a_cm[i] = a_cm[i]/mtot;
    }
    // printf("vcm %.14f %.14f %.14f\n", v_cm[0], v_cm[1], v_cm[2]);
    // printf("acm %.14f %.14f %.14f\n", a_cm[0], a_cm[1], a_cm[2]);

    // Retard position and velocity by one step relative to center of mass??
    for (k=0; k<config.n_bodies; k++) {
        b.x[k] = b.x[k] - v_cm[0]*config.dt;
        b.y[k] = b.y[k] - v_cm[1]*config.dt;
        b.z[k] = b.z[k] - v_cm[2]*config.dt;

        b.vx[k] = b.vx[k] - a_cm[0]*config.dt;
        b.vy[k] = b.vy[k] - a_cm[1]*config.dt;
        b.vz[k] = b.vz[k] - a_cm[2]*config.dt;
    }

    // Increase tidal field
    t_tidal = ((double)iter + 1.) / ((double)config.n_tidal);
    strength = (-2.*t_tidal + 3.)*t_tidal*t_tidal;

    // Find new accelerations
    update_acceleration(config, b, p, f, pot,
                        strength, tnow, &not_firstc);

    // Advance velocity by one step
    step_vel(config, b, config.dt, tnow, tvel);

    // Reset times
    *tvel = 0.5*config.dt;
    *tpos = 0.;
    *tnow = 0.;
}

// void step_system(int iter, Config config, Bodies b, Placeholders p, COMFrame *f,
//                  CPotential *pot, double *tnow, double *tpos, double *tvel) {

//     int not_firstc = 0;
//     double strength = 1.;

//     printf("xyz %.10e %.10e %.10e\n", b.x[99], b.y[99], b.z[99]);
//     printf("vxyz %.10e %.10e %.10e\n\n", b.vx[99], b.vy[99], b.vz[99]);

//     step_pos(config, b, config.dt, tnow, tpos);
//     frame(iter, config, b, f);
//     acc_pot(config, b, p, f, pot,
//             strength, tnow, &not_firstc);
//     step_vel(config, b, config.dt, tnow, tvel);
// }

// void compute_progenitor_energy(Config config, Bodies b, Placeholders p, COMFrame *f,
//                                CPotential *pot, double *tnow, double *tpos, double *tvel) {
//     int k;

//     f.E_tot = 0.;
//     f.E_pot_ext = 0.;
//     f.E_pot_self = 0.;
//     f.E_kin = 0.;

//     // TODO: keep track of external potential separate from BFE potential
//     for (k=0; k<config.n_bodies; k++) {
//               epext=epext+mass(i)*potext(i)
//               epselfg=epselfg+0.5*mass(i)*pot(i)
//               ektot=ektot+0.5*mass(i)*((vx(i))**2+
//         &         (vy(i))**2+(vz(i))**2)
//     }
// }
