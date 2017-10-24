#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <math.h>
#include <Python.h>
#include "potential/src/cpotential.h" // from gala/potential
#include "helpers.h"
#include "scf.h"

void accp_firstc(Config config, Placeholders p) {
    /*
    This code follows the "if (firstc)" block of the original Fortran
    implementation of SCF. This just initializes values for arrays of
    coefficients needed for the basis function expansion.
    */
    int n,l,m,idx;
    double knl, arggam, deltam0;

    p.dblfact[1] = 1.;
    for (l=2; l<=config.lmax; l++) {
        p.dblfact[l] = p.dblfact[l-1]*(2.*l-1.);
    }

    for (n=0; n<=config.nmax; n++) {
        for (l=0; l <= config.lmax; l++) {
            knl = 0.5*n*(n+4.*l+3.)+(l+1.)*(2.*l+1.);

            idx = getIndex2D(n,l,config.lmax+1);
            p.anltilde[idx] = -pow(2.,(8.*l+6.)) * gsl_sf_fact(n)*(n+2.*l+1.5);
            p.anltilde[idx] = p.anltilde[idx] * pow(gsl_sf_gamma(2*l + 1.5), 2);
            p.anltilde[idx] = p.anltilde[idx] / (4.*M_PI*knl*gsl_sf_fact(n+4*l+2));
        }
    }

    for (l=0; l <= config.lmax; l++) {
        p.twoalpha[l] = 2.*(2.*l+1.5);
        for (m=0; m<=l; m++) {
            deltam0 = 2.;
            if (m == 0)
                deltam0 = 1.;

            idx = getIndex2D(l,m,config.lmax+1);
            p.coeflm[idx] = (2.*l+1.) * deltam0 * gsl_sf_fact(l-m) / gsl_sf_fact(l+m);
        }
    }

    for (n=1; n<=config.nmax; n++) {
        p.c3[n] = 1. / (n+1.);
        for (l=0; l<=config.lmax; l++) {
            idx = getIndex2D(n,l,config.lmax+1);
            p.c1[idx] = 2.0*n + p.twoalpha[l];
            p.c2[idx] = n-1.0 + p.twoalpha[l];
        }
    }

    *(p.lskip) = 1;
    if (config.zeroodd || config.zeroeven) {
        *(p.lskip) = 2;
    }

    *(p.lmin) = 0;
    if (config.zeroeven) {
        *(p.lmin) = 1;
    }
}

void accp_bfe(Config config, Bodies b, Placeholders p, int *firstc) {
    /*
    Compute the acceleration and potential energy from the basis function
    expansion (BFE) estimate of the gravitational potential/density of
    particles still bound to the satellite system.

    Parameters
    ----------
    config : Config (struct)
        Struct containing configuration parameters.
    b : Bodies (struct)
        Struct of pointers to arrays that contain information about the mass
        particles (the bodies).
    p : Placeholders (struct)
        Struct of pointers to placeholder arrays used in the BFE calculations.
    firstc : int
        Boolean integer value specifying whether this is the first acceleration
        calculation or not. If so, will call `accp_firstc()` to initialize the
        BFE coefficient / placeholder arrays.
    */

    int j,k,n,l,m, i1,i2;
    double r, costh, phi, xi;
    double un, unm1, plm1m, plm2m;
    double temp3, temp4, temp5, temp6, ttemp5;
    double ar, ath, aphi, cosp, sinp, phinltil, sinth;
    double clm, dlm, elm, flm;
    int lmax = config.lmax;
    int nmax = config.nmax;
    double cosmphi[lmax+1], sinmphi[lmax+1];

    if (*firstc) {
        accp_firstc(config, p);
        *firstc = 0;
    }

    // zero out the coefficients
    for (l=0; l<=lmax; l++) { // zero all out
        for (m=0; m<=l; m++) {
            for (n=0; n<=nmax; n++) {
                i1 = getIndex3D(n,l,m,lmax+1,lmax+1);
                p.sinsum[i1] = 0.;
                p.cossum[i1] = 0.;
            }
        }
    }

    // This loop computes the BFE coefficients for all bound particles
    for (k=0; k<config.n_bodies; k++) {
        // printf("%d\n", k);

        if (b.ibound[k] > 0) { // skip unbound particles
            r = sqrt(b.x[k]*b.x[k] + b.y[k]*b.y[k] + b.z[k]*b.z[k]);
            costh = b.z[k] / r;
            phi = atan2(b.y[k], b.x[k]);
            xi = (r - 1.) / (r + 1.);

            // precompute all cos(m*phi), sin(m*phi)
            for (m=0; m<(lmax+1); m++) {
                cosmphi[m] = cos(m*phi);
                sinmphi[m] = sin(m*phi);
            }

            for (l=0; l<=lmax; l++) { // its ok to compute all
                p.ultrasp[getIndex2D(0,l,lmax+1)] = 1.;
                p.ultrasp[getIndex2D(1,l,lmax+1)] = p.twoalpha[l]*xi;

                un = p.ultrasp[getIndex2D(1,l,lmax+1)];
                unm1 = 1.0;

                for (n=1; n<nmax; n++) {
                    i1 = getIndex2D(n+1,l,lmax+1);
                    i2 = getIndex2D(n,l,lmax+1);
                    p.ultrasp[i1] = (p.c1[i2]*xi*un-p.c2[i2]*unm1)*p.c3[n];
                    unm1 = un;
                    un = p.ultrasp[i1];
                }

                for (n=0; n<=nmax; n++) {
                    i1 = getIndex2D(n,l,lmax+1);
                    p.ultraspt[i1] = p.ultrasp[i1] * p.anltilde[i1];
                }
            }

            for (m=0; m<=lmax; m++) {
                i1 = getIndex2D(m,m,lmax+1);
                p.plm[i1] = 1.0;
                if (m > 0) {
                    p.plm[i1] = pow(-1.,m) * p.dblfact[m] * pow(sqrt(1.-costh*costh), m);
                }

                plm1m = p.plm[i1];
                plm2m = 0.0;

                for (l=m+1; l<=lmax; l++) {
                    i2 = getIndex2D(l,m,lmax+1);
                    p.plm[i2] = (costh*(2.*l-1.)*plm1m - (l+m-1.)*plm2m) / (l-m);
                    plm2m = plm1m;
                    plm1m = p.plm[i2];
                }
            }

            for (l=(*(p.lmin)); l<=lmax; l=l+(*(p.lskip))) {
                temp5 = pow(r,l) / pow(1.+r,2*l+1) * b.mass[k];
                // printf("temp5 %f %f %d %f\n", temp5, r, l, mass[k]);

                for (m=0; m<=l; m++) {
                    i1 = getIndex2D(l,m,lmax+1);
                    ttemp5 = temp5 * p.plm[i1] * p.coeflm[i1];
                    temp3 = ttemp5 * sinmphi[m];
                    temp4 = ttemp5 * cosmphi[m];

                    for (n=0; n<=nmax; n++) {
                        i1 = getIndex2D(n,l,lmax+1);
                        i2 = getIndex3D(n,l,m,lmax+1,lmax+1);
                        p.sinsum[i2] = p.sinsum[i2] + temp3*p.ultraspt[i1];
                        p.cossum[i2] = p.cossum[i2] + temp4*p.ultraspt[i1];
                    }
                }
            }
        }
    }

    // This loop computes the acceleration and potential at each particle given the BFE coeffs
    for (k=0; k<config.n_bodies; k++) {
        r = sqrt(b.x[k]*b.x[k] + b.y[k]*b.y[k] + b.z[k]*b.z[k]);
        costh = b.z[k] / r;
        phi = atan2(b.y[k], b.x[k]);
        xi = (r - 1.) / (r + 1.);

        // precompute all cos(m*phi), sin(m*phi)
        for (m=0; m<(lmax+1); m++) {
            cosmphi[m] = cos(m*phi);
            sinmphi[m] = sin(m*phi);
        }

        // Zero out potential and accelerations
        b.Epot_bfe[k] = 0.;
        ar = 0.;
        ath = 0.;
        aphi = 0.;

        for (l=0; l<=lmax; l++) {
            p.ultrasp[getIndex2D(0,l,lmax+1)] = 1.;
            p.ultrasp[getIndex2D(1,l,lmax+1)] = p.twoalpha[l]*xi;
            p.ultrasp1[getIndex2D(0,l,lmax+1)] = 0.;
            p.ultrasp1[getIndex2D(1,l,lmax+1)] = 1.;

            un = p.ultrasp[getIndex2D(1,l,lmax+1)];
            unm1 = 1.;

            for (n=1; n<nmax; n++) {
                i1 = getIndex2D(n+1,l,lmax+1);
                i2 = getIndex2D(n,l,lmax+1);
                p.ultrasp[i1] = (p.c1[i2]*xi*un - p.c2[i2]*unm1) * p.c3[n];
                unm1 = un;
                un = p.ultrasp[i1];
                p.ultrasp1[i1] = ((p.twoalpha[l]+(n+1)-1.)*unm1-(n+1)*xi*p.ultrasp[i1]) / (p.twoalpha[l]*(1.-xi*xi));
            }
        }

        for (m=0; m<=lmax; m++) {
            i1 = getIndex2D(m,m,lmax+1);
            p.plm[i1] = 1.0;
            if (m > 0) {
                p.plm[i1] = pow(-1.,m) * p.dblfact[m] * pow(sqrt(1.-costh*costh), m);
            }

            plm1m = p.plm[i1];
            plm2m = 0.0;

            for (l=m+1; l<=lmax; l++) {
                i2 = getIndex2D(l,m,lmax+1);
                p.plm[i2] = (costh*(2.*l-1.)*plm1m - (l+m-1.)*plm2m) / (l-m);
                plm2m = plm1m;
                plm1m = p.plm[i2];
            }
        }

        p.dplm[0,0] = 0.;

        for (l=1; l<=lmax; l++) {
            for (m=0; m<=l; m++) {
                i1 = getIndex2D(l,m,lmax+1);
                if (l == m) {
                    p.dplm[i1]=l*costh*p.plm[i1]/(costh*costh-1.0);
                } else {
                    i2 = getIndex2D(l-1,m,lmax+1);
                    p.dplm[i1]=(l*costh*p.plm[i1]-(l+m)*p.plm[i2]) / (costh*costh-1.0);
                }
            }
        }

        for (l=(*(p.lmin)); l<=lmax; l=l+(*(p.lskip))) {
            temp3 = 0.;
            temp4 = 0.;
            temp5 = 0.;
            temp6 = 0.;

            for (m=0; m<=l; m++) {
                clm = 0.;
                dlm = 0.;
                elm = 0.;
                flm = 0.;
                for (n=0; n<=nmax; n++) {
                    i1 = getIndex2D(n,l,lmax+1);
                    i2 = getIndex3D(n,l,m,lmax+1,lmax+1);

                    clm = clm + p.ultrasp[i1]*p.cossum[i2];
                    dlm = dlm + p.ultrasp[i1]*p.sinsum[i2];
                    elm = elm + p.ultrasp1[i1]*p.cossum[i2];
                    flm = flm + p.ultrasp1[i1]*p.sinsum[i2];
                }

                i1 = getIndex2D(l,m,lmax+1);
                temp3 = temp3 + p.plm[i1]*(clm*cosmphi[m]+dlm*sinmphi[m]);
                temp4 = temp4 - p.plm[i1]*(elm*cosmphi[m]+flm*sinmphi[m]);
                temp5 = temp5 - p.dplm[i1]*(clm*cosmphi[m]+dlm*sinmphi[m]);
                temp6 = temp6 - m*p.plm[i1]*(dlm*cosmphi[m]-clm*sinmphi[m]);
            }

            phinltil = pow(r,l) / pow(1.+r, 2*l+1);
            b.Epot_bfe[k] = b.Epot_bfe[k] + temp3*phinltil;
            ar = ar + phinltil*(-temp3*(l/r-(2.*l+1.)/(1.+r)) + temp4*4.*(2.*l+1.5)/pow(1.+r,2));
            ath = ath + temp5*phinltil;
            aphi = aphi + temp6*phinltil;
        }

        cosp = cos(phi);
        sinp = sin(phi);

        sinth = sqrt(1.-costh*costh);
        ath = -sinth*ath/r;
        aphi = aphi/(r*sinth);

        b.ax[k] = config.G*(sinth*cosp*ar + costh*cosp*ath - sinp*aphi);
        b.ay[k] = config.G*(sinth*sinp*ar + costh*sinp*ath + cosp*aphi);
        b.az[k] = config.G*(costh*ar - sinth*ath);
        b.Epot_bfe[k] = b.Epot_bfe[k]*config.G;
    }

}

void accp_external(Config config, Bodies b, COMFrame *f,
                   CPotential *pot, double strength, double *tnow) {

    int j, k;
    double grad[3], q[3];

    for (k=0; k<config.n_bodies; k++) {
        q[0] = (b.x[k] + (f->x));
        q[1] = (b.y[k] + (f->y));
        q[2] = (b.z[k] + (f->z));

        // Compute external potential gradient
        c_gradient(pot, *tnow, &q[0], &grad[0]);
        b.Epot_ext[k] = c_potential(pot, *tnow, &q[0]);

        b.ax[k] = b.ax[k] - strength*grad[0];
        b.ay[k] = b.ay[k] - strength*grad[1];
        b.az[k] = b.az[k] - strength*grad[2];
    }
}

void acc_pot(Config config, Bodies b, Placeholders p, COMFrame *f,
             CPotential *pot, double extern_strength, double *tnow, int *firstc) {
    /*
    Compute the total acceleration and potential energy for each N body.

    Parameters
    ----------
    config : Config (struct)
        Struct containing configuration parameters.
    b : Bodies (struct)
        Struct of pointers to arrays that contain information about the mass
        particles (the bodies).
    p : Placeholders (struct)
        Struct of pointers to placeholder arrays used in the BFE calculations.
    extern_strength : double
        The strength of the external potential (normalized to the range (0,1)).
        Used by `tidal_start()` to slowly turn on the external field.
    firstc : int
        Boolean integer value specifying whether this is the first acceleration
        calculation or not. If so, will call `accp_firstc()` to initialize the
        BFE coefficient / placeholder arrays.
    */
    int j,k;

    if (config.selfgravitating) {
        accp_bfe(config, b, p, firstc);
        accp_external(config, b, f, pot, extern_strength, tnow);
    } else {
        for (k=0; k<config.n_bodies; k++) {
            b.ax[k] = 0.;
            b.ay[k] = 0.;
            b.az[k] = 0.;
            b.Epot_ext[k] = 0.;
            b.Epot_bfe[k] = 0.;
        }
        accp_external(config, b, f, pot, extern_strength, tnow);
    }

}

void frame(int iter, Config config, Bodies b, COMFrame *f) {
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

    // take the most bound particles (?)
    int nend = config.n_bodies / 100;
    if (nend < 128) nend = 128;
    int i,j,k;

    double mred = 0.;
    double xyz_min[3];
    xyz_min[0] = 0.;
    xyz_min[1] = 0.;
    xyz_min[2] = 0.;

    // TODO / Note: n_recenter is currently disabled
    // if ((iter == 0) || ((iter % config.n_recenter) == 0)) {
    //     indexx(config.n_bodies, b.pot, (f->pot_idx));
    // }

    // for (i=0; i<=6; i++) {
    //     printf("rank %d %d %.14e\n", i, (f->pot_idx)[i], b.pot[(f->pot_idx)[i]]);
    // }

    for (i=0; i<nend; i++) {
        k = (f->pot_idx)[i];
        xyz_min[0] = xyz_min[0] + b.mass[k]*b.x[k];
        xyz_min[1] = xyz_min[1] + b.mass[k]*b.y[k];
        xyz_min[2] = xyz_min[2] + b.mass[k]*b.z[k];
        mred = mred + b.mass[k];
    }

    xyz_min[0] = xyz_min[0] / mred;
    xyz_min[1] = xyz_min[1] / mred;
    xyz_min[2] = xyz_min[2] / mred;

    // Update frame and shift to center on the minimum of the potential
    (f->x) = (f->x) + xyz_min[0];
    (f->y) = (f->y) + xyz_min[1];
    (f->z) = (f->z) + xyz_min[2];

    for (k=0; k<config.n_bodies; k++) {
        b.x[k] = b.x[k] - xyz_min[0];
        b.y[k] = b.y[k] - xyz_min[1];
        b.z[k] = b.z[k] - xyz_min[2];
    }

    // find velocity frame
    (f->vx) = 0.;
    (f->vy) = 0.;
    (f->vz) = 0.;
    for (i=0; i<nend; i++) {
        k = (f->pot_idx)[i];
        // printf("%d\n", k);
        (f->vx) = (f->vx) + b.mass[k]*b.vx[k];
        (f->vy) = (f->vy) + b.mass[k]*b.vy[k];
        (f->vz) = (f->vz) + b.mass[k]*b.vz[k];
    }
    (f->vx) = (f->vx) / mred;
    (f->vy) = (f->vy) / mred;
    (f->vz) = (f->vz) / mred;

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
                // printf("unbound %d %.8e %.8e %.8e\n", k+1, b.Ekin[k], b.Epot_bfe, b.Epot_ext[k]);
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
        acc_pot(config, b, p, f,
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
    acc_pot(config, b, p, f, pot,
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
