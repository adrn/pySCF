#include "gsl/gsl_sf_gamma.h"
#include "gsl/gsl_sf_legendre.h"
#include "gsl/gsl_sf_gegenbauer.h"
#include <math.h>
#include "helpers.h"

int getIndex2D(int row, int col, int ncol) {
    return row*ncol + col;
}

int getIndex3D(int row, int col, int dep, int ncol, int ndep) {
    return (row*ncol + col)*ndep + dep;
}

void accp_firstc(int nmax, int lmax, int zeroodd, int zeroeven, // INPUT
                 double *dblfact, // OUTPUT: length = lmax+1
                 double *twoalpha, // OUTPUT: length = lmax+1
                 double *anltilde, // OUTPUT: length = (nmax+1)*(lmax+1)
                 double *coeflm, // OUTPUT: length = (lmax+1)*(lmax+1)
                 int *lmin, int *lskip, // OUTPUT: integers
                 double *c1, // OUTPUT: length = (nmax+1)*(lmax+1)
                 double *c2, // OUTPUT: length = (nmax+1)*(lmax+1)
                 double *c3) // OUTPUT: length = (nmax+1)
{
    /*
    This code follows the "if (firstc)" block of the original Fortran
    implementation of SCF.

    Will call with:

    int lmin, lskip;
    accp_firstc(..., &dblfact[0], ..., &lmin, &lskip, ...)
    */
    int n,l,m,idx;
    double knl, arggam, deltam0;

    dblfact[1] = 1.;
    for (l=2; l <= lmax; l++) {
        dblfact[l] = dblfact[l-1]*(2.*l-1.);
    }

    for (n=0; n<=nmax; n++) {
        for (l=0; l <= lmax; l++) {
            knl = 0.5*n*(n+4.*l+3.)+(l+1.)*(2.*l+1.);

            idx = getIndex2D(n,l,lmax+1);
            anltilde[idx] = pow(-2.,(8.*l+6.)) * gsl_sf_fact(n)*(n+2.*l+1.5);
            anltilde[idx] = anltilde[idx] * pow(gsl_sf_gamma(2*l + 1.5), 2);
            anltilde[idx] = anltilde[idx] / (4.*M_PI*knl*gsl_sf_fact(n+4*l+2));
        }
    }

    for (l=0; l <= lmax; l++) {
        twoalpha[l] = 2.*(2.*l+1.5);
        for (m=0; m<=l; m++) {
            deltam0 = 2.;
            if (m == 0)
                deltam0 = 1.;

            idx = getIndex2D(l,m,lmax+1);
            coeflm[idx] = (2.*l+1.) * deltam0 * gsl_sf_fact(l-m) / gsl_sf_fact(l+m);
        }
    }

    for (n=1; n<=nmax; n++) {
        c3[n] = 1. / (n+1.);
        for (l=0; l<=lmax; l++) {
            idx = getIndex2D(n,l,lmax+1);
            c1[idx] = 2.0*n + twoalpha[l];
            c2[idx] = n-1.0 + twoalpha[l];
        }
    }

    *lskip = 1;
    if (zeroodd || zeroeven) {
        *lskip = 2;
    }

    *lmin = 0;
    if (zeroeven) {
        *lmin = 1;
    }

}

void accp_LH(int nbodies, double *xyz, double *mass, int *ibound, // INPUT
             int nmax, int lmax, int zeroodd, int zeroeven, // INPUT
             double *sinsum, double *cossum, // INPUT: length = (nmax+1)*(lmax+1)*(lmax+1) (to avoid re-defining)
             double G, int *firstc, // INPUT
             double *dblfact, double *twoalpha, double *anltilde, double *coeflm,
             int *lmin, int *lskip,
             double *c1, double *c2, double *c3,
             double *pot, // OUTPUT: length = nbodies
             double *acc) // OUTPUT: length = 3*nbodies
{
    /*
    */

    int j,k,n,l,m, i1,i2;
    double r, costh, phi, xi;
    double un, unm1, plm1m, plm2m;
    double temp3, temp4, temp5, temp6, ttemp5;
    double ar, ath, aphi, cosp, sinp, phinltil, sinth;
    double clm, dlm, elm, flm;

    // TODO: should I define all of this outside?
    double cosmphi[lmax+1], sinmphi[lmax+1];
    double ultrasp[(nmax+1)*(lmax+1)], ultraspt[(nmax+1)*(lmax+1)], ultrasp1[(nmax+1)*(lmax+1)];
    double plm[(lmax+1)*(lmax+1)], dplm[(lmax+1)*(lmax+1)];

    printf("firstc %d\n", *firstc);
    if (*firstc) {
        accp_firstc(nmax, lmax, zeroodd, zeroeven,
                    &dblfact[0], &twoalpha[0], &anltilde[0,0], &coeflm[0,0],
                    lmin, lskip, &c1[0,0], &c2[0,0], &c3[0]);
        *firstc = 0;
    }

    // printf("lmin lskip %d %d\n", *lmin, *lskip);
    // printf("dblfact %f %f %f %f\n", dblfact[0], dblfact[1], dblfact[2], dblfact[3]);
    // printf("twoalpha %f %f %f %f\n", twoalpha[0], twoalpha[1], twoalpha[2], twoalpha[3]);
    // printf("anltilde %f %f %f %f\n", anltilde[0], anltilde[1], anltilde[2], anltilde[3]);
    // printf("coeflm %f %f %f %f\n", coeflm[0], coeflm[1], coeflm[2], coeflm[3]);
    // printf("c1 %f %f %f %f %f\n", c1[0], c1[1], c1[2], c1[3], c1[4]);
    // printf("c2 %f %f %f %f %f\n", c2[0], c2[1], c2[2], c2[3], c2[4]);
    // printf("c3 %f %f %f %f\n", c3[0], c3[1], c3[2], c3[3]);

    // zero out the coefficients
    for (l=0; l<=lmax; l++) {
        for (m=0; m<=l; m++) {
            for (n=0; n<=nmax; n++) {
                i1 = getIndex3D(n,l,m,lmax+1,lmax+1);
                sinsum[i1] = 0.;
                cossum[i1] = 0.;
            }
        }
    }

    // This loop computes the BFE coefficients for all bound particles
    for (k=0; k<nbodies; k++) {
        // printf("%d\n", k);

        if (ibound[k] > 0) { // skip unbound particles
            j = 3*k; // x,y,z in same 2D array

            r = sqrt(xyz[j]*xyz[j] + xyz[j+1]*xyz[j+1] + xyz[j+2]*xyz[j+2]);
            costh = xyz[j+2] / r;
            phi = atan2(xyz[j+1], xyz[j+0]);
            xi = (r - 1.) / (r + 1.);

            // precompute all cos(m*phi), sin(m*phi)
            for (m=0; m<(lmax+1); m++) {
                cosmphi[m] = cos(m*phi);
                sinmphi[m] = sin(m*phi);
            }
            // printf("cosmphi %f %f %f %f\n", cosmphi[0], cosmphi[1], cosmphi[2], cosmphi[3]);
            // printf("sinmphi %f %f %f %f\n", sinmphi[0], sinmphi[1], sinmphi[2], sinmphi[3]);

            for (l=0; l<=lmax; l++) {
                ultrasp[getIndex2D(0,l,lmax+1)] = 1.;
                ultrasp[getIndex2D(1,l,lmax+1)] = twoalpha[l]*xi;

                un = ultrasp[getIndex2D(1,l,lmax+1)];
                unm1 = 1.0;

                for (n=1; n<nmax; n++) {
                    i1 = getIndex2D(n+1,l,lmax+1);
                    i2 = getIndex2D(n,l,lmax+1);
                    ultrasp[i1] = (c1[i2]*xi*un-c2[i2]*unm1)*c3[n];
                    unm1 = un;
                    un = ultrasp[i1];
                }

                for (n=0; n<=nmax; n++) {
                    i1 = getIndex2D(n,l,lmax+1);
                    ultraspt[i1] = ultrasp[i1] * anltilde[i1];
                }
            }

            // printf("ultraspt %f %f %f %f\n", ultraspt[0], ultraspt[1], ultraspt[2], ultraspt[3]);
            // printf("ultrasp %f %f %f %f\n", ultrasp[0], ultrasp[1], ultrasp[2], ultrasp[3]);

            for (m=0; m<=lmax; m++) {
                i1 = getIndex2D(m,m,lmax+1);
                plm[i1] = 1.0;
                if (m > 0) {
                    plm[i1] = pow(-1.,m) * dblfact[m] * pow(sqrt(1.-costh*costh), m);
                }

                plm1m = plm[i1];
                plm2m = 0.0;

                for (l=m+1; l<=lmax; l++) {
                    i2 = getIndex2D(l,m,lmax+1);
                    plm[i2] = (costh*(2.*l-1.)*plm1m - (l+m-1.)*plm2m) / (l-m);
                    plm2m = plm1m;
                    plm1m = plm[i2];
                }
            }
            // printf("plm %f %f %f %f\n", plm[0], plm[1], plm[2], plm[3]);

            for (l=(*lmin); l<=lmax; l=l+(*lskip)) {
                temp5 = pow(r,l) / pow(1.+r,2*l+1) * mass[k];
                // printf("temp5 %f\n", temp5);

                for (m=0; m<=l; m++) {
                    i1 = getIndex2D(l,m,lmax+1);
                    ttemp5 = temp5*plm[i1]*coeflm[i1];
                    temp3 = ttemp5 * sinmphi[m];
                    temp4 = ttemp5 * cosmphi[m];

                    for (n=0; n<=nmax; n++) {
                        i1 = getIndex3D(n,l,m,lmax+1,lmax+1);
                        i2 = getIndex2D(n,l,lmax+1);
                        sinsum[i1] = sinsum[i1] + temp3*ultraspt[i2];
                        cossum[i1] = cossum[i1] + temp4*ultraspt[i2];
                    }
                }
            }
        }
    }

    // This loop computes the acceleration and potential at each particle given the BFE coeffs
    for (k=0; k<nbodies; k++) {
        j = 3*k; // x,y,z in same 2D array

        r = sqrt(xyz[j]*xyz[j] + xyz[j+1]*xyz[j+1] + xyz[j+2]*xyz[j+2]);
        costh = xyz[j+2] / r;
        phi = atan2(xyz[j+1], xyz[j+0]);
        xi = (r - 1.) / (r + 1.);

        // precompute all cos(m*phi), sin(m*phi)
        for (m=0; m<(lmax+1); m++) {
            cosmphi[m] = cos(m*phi);
            sinmphi[m] = sin(m*phi);
        }

        // Zero out potential and accelerations
        pot[k] = 0.;
        ar = 0.;
        ath = 0.;
        aphi = 0.;

        for (l=0; l<=lmax; l++) {
            ultrasp[0,l] = 1.;
            ultrasp[1,l] = twoalpha[l]*xi;
            ultrasp1[0,l] = 0.;
            ultrasp1[1,l] = 1.;

            un = ultrasp[1,l];
            unm1 = 1.;

            for (n=1; n<nmax; n++) {
                ultrasp[n+1,l] = (c1[n,l]*xi*un - c2[n,l]*unm1)*c3[n];
                unm1 = un;
                un = ultrasp[n+1,l];
                ultrasp1[n+1,l] = ((twoalpha[l]+(n+1)-1.)*unm1-(n+1)*xi*ultrasp[n+1,l]) / (twoalpha[l]*(1.-xi*xi));
            }
        }

        for (m=0; m<=lmax; m++) {
            i1 = getIndex2D(m,m,lmax+1);
            plm[i1] = 1.0;
            if (m > 0) {
                plm[i1] = pow(-1.,m) * dblfact[m] * pow(sqrt(1.-costh*costh), m);
            }

            plm1m = plm[i1];
            plm2m = 0.0;

            for (l=m+1; l<=lmax; l++) {
                i2 = getIndex2D(l,m,lmax+1);
                plm[i2] = (costh*(2.*l-1.)*plm1m - (l+m-1.)*plm2m) / (l-m);
                plm2m = plm1m;
                plm1m = plm[i2];
            }
        }

        dplm[0,0] = 0.;

        for (l=1; l<=lmax; l++) {
            for (m=0; m<=l; m++) {
                i1 = getIndex2D(l,m,lmax+1);
                if (l == m) {
                    dplm[i1]=l*costh*plm[i1]/(costh*costh-1.0);
                } else {
                    i2 = getIndex2D(l-1,m,lmax+1);
                    dplm[i1]=(l*costh*plm[i1]-(l+m)*plm[i2]) / (costh*costh-1.0);
                }
            }
        }

        for (l=(*lmin); l<=lmax; l=l+(*lskip)) {
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
                    clm = clm + ultrasp[i1]*cossum[i2];
                    dlm = dlm + ultrasp[i1]*sinsum[i2];
                    elm = elm + ultrasp1[i1]*cossum[i2];
                    flm = flm + ultrasp1[i1]*sinsum[i2];
                }

                i1 = getIndex2D(l,m,lmax+1);
                temp3 = temp3 + plm[i1]*(clm*cosmphi[m]+dlm*sinmphi[m]);
                temp4 = temp4 - plm[i1]*(elm*cosmphi[m]+flm*sinmphi[m]);
                temp5 = temp5 - dplm[i1]*(clm*cosmphi[m]+dlm*sinmphi[m]);
                temp6 = temp6 - m*plm[i1]*(dlm*cosmphi[m]-clm*sinmphi[m]);
            }

            phinltil = pow(r,l) / pow(1.+r, 2*l+1);
            pot[k] = pot[k] + temp3*phinltil;
            ar = ar + phinltil*(-temp3*(l/r-(2.*l+1.)/(1.+r)) + temp4*4.*(2.*l+1.5)/pow(1.+r,2));
            ath = ath + temp5*phinltil;
            aphi = aphi + temp6*phinltil;
        }

        cosp = cos(phi);
        sinp = sin(phi);

        sinth = sqrt(1.-costh*costh);
        ath = -sinth*ath/r;
        aphi = aphi/(r*sinth);

        acc[j+0] = G*(sinth*cosp*ar + costh*cosp*ath - sinp*aphi);
        acc[j+1] = G*(sinth*sinp*ar + costh*sinp*ath + cosp*aphi);
        acc[j+2] = G*(costh*ar - sinth*ath);
        pot[k] = pot[k]*G;
    }

}

void accp_external(int nbodies, double *xyz,
                   double *pot, double *acc) {
    int k;
    for (k=0; k<nbodies; k++) {
        // TODO:
    }
}

void acc_pot(int selfgrav,
             int nbodies, double *xyz, double *mass, int *ibound, // INPUT
             int nmax, int lmax, int zeroodd, int zeroeven, // INPUT
             double *sinsum, double *cossum, // INPUT: length = (nmax+1)*(lmax+1)*(lmax+1) (to avoid re-defining)
             double G, int *firstc, // INPUT
             double *dblfact, double *twoalpha, double *anltilde, double *coeflm,
             int *lmin, int *lskip,
             double *c1, double *c2, double *c3,
             double *pot, // OUTPUT: length = nbodies
             double *acc) // OUTPUT: length = 3*nbodies
{
    int j,k;

    if (selfgrav) {
        accp_LH(nbodies, xyz, mass, ibound,
                nmax, lmax, zeroodd, zeroeven,
                sinsum, cossum, G, firstc,
                dblfact, twoalpha, anltilde, coeflm, lmin, lskip,
                c1, c2, c3, pot, acc);

        accp_external(nbodies, xyz, pot, acc);

    } else {
        for (k=0; k<nbodies; k++) {
            j = 3*k;
            acc[j+0] = 0.;
            acc[j+1] = 0.;
            acc[j+2] = 0.;
            pot[k] = 0.;
        }
        accp_external(nbodies, xyz, pot, acc);
    }
}

void frame(int iter, int n_recenter,
           int nbodies, double *xyz, double *vxyz, double *mass,
           double *pot) {
    /*
    Shift the phase-space coordinates to be centered on the minimum potential.
    The initial value is the input position and velocity of the progenitor system.

    Parameters
    ----------
    iter : int
        The index of the current iteration (starting from 0).
    n_recenter : int
        After how many steps should we recenter the ...
    */
    double idx[nbodies];
    int nend = nbodies / 100; // take the most bound particles (?)

    double mred;
    double xyz_min[3];
    memset(xyz_min, 0, 3*sizeof(double));

    if ((iter == 0) || ((iter % n_recenter) == 0)) {
        indexx(nbodies, pot, idx);
    }



}

void tidal_start(double *xyz, double *vxyz, double *mass) {

}
