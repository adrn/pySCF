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
from cython.parallel import parallel, prange

from libc.math cimport sin, cos, atan2, sqrt, M_PI
from cython_gsl cimport gsl_sf_fact, gsl_sf_gamma

from gala.potential.potential.cpotential cimport (CPotential,
                                                  c_potential,
                                                  c_gradient)

from structs cimport Config, Placeholders, Bodies, COMFrame

# needed for some reason
cdef extern from "src/helpers.h":
    void indexx(int n, double *arrin, int *indx) nogil
    int getIndex2D(int row, int col, int ncol) nogil
    int getIndex3D(int row, int col, int dep, int ncol, int ndep) nogil

# ----------------------------------------------------------------------------

cdef void internal_bfe_init(Config config, Placeholders p):
    """
    This code follows the "if (firstc)" block of the original Fortran
    implementation of SCF. This just initializes values for arrays of
    coefficients needed for the basis function expansion.
    """
    cdef:
        int n,l,m,idx
        double knl, arggam, deltam0

    p.dblfact[1] = 1.
    for l in range(2, config.lmax+1):
        p.dblfact[l] = p.dblfact[l-1]*(2.*l-1.)

    for n in range(config.nmax+1):
        for l in range(0, config.lmax+1):
            knl = 0.5*n*(n+4.*l+3.)+(l+1.)*(2.*l+1.)

            idx = getIndex2D(n,l,config.lmax+1)
            p.anltilde[idx] = -pow(2.,(8.*l+6.)) * gsl_sf_fact(n)*(n+2.*l+1.5)
            p.anltilde[idx] = p.anltilde[idx] * pow(gsl_sf_gamma(2*l + 1.5), 2)
            p.anltilde[idx] = p.anltilde[idx] / (4.*M_PI*knl*gsl_sf_fact(n+4*l+2))

    for l in range(0, config.lmax+1):
        p.twoalpha[l] = 2.*(2.*l+1.5)
        for m in range(0, l+1):
            deltam0 = 2.
            if m == 0:
                deltam0 = 1.

            idx = getIndex2D(l, m, config.lmax+1)
            p.coeflm[idx] = (2.*l+1.) * deltam0 * gsl_sf_fact(l-m) / gsl_sf_fact(l+m)

    for n in range(1, config.nmax+1):
        p.c3[n] = 1. / (n+1.)
        for l in range(0, config.lmax+1):
            idx = getIndex2D(n, l, config.lmax+1)
            p.c1[idx] = 2.0*n + p.twoalpha[l]
            p.c2[idx] = n-1.0 + p.twoalpha[l]

    p.lskip[0] = 1
    if config.zeroodd or config.zeroeven:
        p.lskip[0] = 2

    p.lmin[0] = 0
    if config.zeroeven:
        p.lmin[0] = 1

cdef void internal_bfe_field(Config config, Bodies b, Placeholders p, COMFrame *f,
                             int *firstc):
    """
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
        calculation or not. If so, will call `internal_bfe_init()` to initialize
        the BFE coefficient / placeholder arrays.
    """

    cdef:
        int j,k,n,l,m, i1,i2
        double r, costh, phi, xi
        double un, unm1, plm1m, plm2m
        double temp3, temp4, temp5, temp6, ttemp5
        double ar, ath, aphi, cosp, sinp, phinltil, sinth
        double clm, dlm, elm, flm
        double xx, yy, zz
        int lmax = config.lmax
        int nmax = config.nmax
        # double cosmphi[lmax+1], sinmphi[lmax+1]
        double[::1] cosmphi = np.zeros(lmax+1, dtype=np.float64)
        double[::1] sinmphi = np.zeros(lmax+1, dtype=np.float64)

        # HACK: because Cython doesn't support step in range()
        int count

        # dereference
        int lmin
        int lskip

    if firstc[0]:
        internal_bfe_init(config, p)
        firstc[0] = 0

    # modified inside internal_bfe_init(), so definitions must be here
    lmin = p.lmin[0]
    lskip = p.lskip[0]

    # zero out the coefficients
    for l in range(0, lmax+1):
        for m in range(0, l+1):
            for n in range(0, nmax+1):
                i1 = getIndex3D(n,l,m,lmax+1,lmax+1)
                p.sinsum[i1] = 0.
                p.cossum[i1] = 0.

    # This loop computes the BFE coefficients for all bound particles.
    for k in range(0, config.n_bodies):
        if b.ibound[k] > 0: # skip unbound particles
            xx = b.x[k] - f.x
            yy = b.y[k] - f.y
            zz = b.z[k] - f.z

            r = sqrt(xx*xx + yy*yy + zz*zz)
            costh = zz / r
            phi = atan2(yy, xx)
            xi = (r - 1.) / (r + 1.)

            # precompute all cos(m*phi), sin(m*phi)
            for m in range(0, lmax+1):
                cosmphi[m] = cos(m*phi)
                sinmphi[m] = sin(m*phi)

            for l in range(0, lmax+1): # its ok to compute all
                p.ultrasp[getIndex2D(0,l,lmax+1)] = 1.
                p.ultrasp[getIndex2D(1,l,lmax+1)] = p.twoalpha[l]*xi

                un = p.ultrasp[getIndex2D(1,l,lmax+1)]
                unm1 = 1.0

                for n in range(1, nmax):
                    i1 = getIndex2D(n+1,l,lmax+1)
                    i2 = getIndex2D(n,l,lmax+1)
                    p.ultrasp[i1] = (p.c1[i2]*xi*un-p.c2[i2]*unm1)*p.c3[n]
                    unm1 = un
                    un = p.ultrasp[i1]

                for n in range(0, nmax+1):
                    i1 = getIndex2D(n,l,lmax+1)
                    p.ultraspt[i1] = p.ultrasp[i1] * p.anltilde[i1]

            for m in range(0, lmax+1):
                i1 = getIndex2D(m,m,lmax+1)
                p.plm[i1] = 1.0
                if m > 0:
                    p.plm[i1] = (-1)**m * p.dblfact[m] * (1.-costh*costh)**(m/2.)

                plm1m = p.plm[i1]
                plm2m = 0.0

                for l in range(m+1, lmax+1):
                    i2 = getIndex2D(l,m,lmax+1)
                    p.plm[i2] = (costh*(2.*l-1.)*plm1m - (l+m-1.)*plm2m) / (l-m)
                    plm2m = plm1m
                    plm1m = p.plm[i2]

            # HACK: Cython doesn't support step in range
            count = (lmax-lmin+lskip) // lskip
            # for l in range(lmin, lmax+1, lskip):
            for l in range(count):
                l = lmin + lskip*l
                temp5 = r**l / (1.+r)**(2*l+1) * b.mass[k]

                for m in range(0, l+1):
                    i1 = getIndex2D(l,m,lmax+1)
                    ttemp5 = temp5 * p.plm[i1] * p.coeflm[i1]
                    temp3 = ttemp5 * sinmphi[m]
                    temp4 = ttemp5 * cosmphi[m]

                    for n in range(0, nmax+1):
                        i1 = getIndex2D(n,l,lmax+1)
                        i2 = getIndex3D(n,l,m,lmax+1,lmax+1)
                        p.sinsum[i2] = p.sinsum[i2] + temp3*p.ultraspt[i1]
                        p.cossum[i2] = p.cossum[i2] + temp4*p.ultraspt[i1]

    # This loop computes the acceleration and potential at each particle given
    # the BFE coeffs.
    with nogil, parallel():
        # for k in range(0, config.n_bodies):
        for k in prange(config.n_bodies, schedule='guided'):
            xx = b.x[k] - f.x
            yy = b.y[k] - f.y
            zz = b.z[k] - f.z

            r = sqrt(xx*xx + yy*yy + zz*zz)
            costh = zz / r
            phi = atan2(yy, xx)
            xi = (r - 1.) / (r + 1.)

            # precompute all cos(m*phi), sin(m*phi)
            # TODO: if we parallelize, can't do this...
            for m in range(0, lmax+1):
                cosmphi[m] = cos(m*phi)
                sinmphi[m] = sin(m*phi)

            # Zero out potential and accelerations
            b.Epot_bfe[k] = 0.
            ar = 0.
            ath = 0.
            aphi = 0.

            for l in range(0, lmax+1):
                p.ultrasp[getIndex2D(0,l,lmax+1)] = 1.
                p.ultrasp[getIndex2D(1,l,lmax+1)] = p.twoalpha[l]*xi
                p.ultrasp1[getIndex2D(0,l,lmax+1)] = 0.
                p.ultrasp1[getIndex2D(1,l,lmax+1)] = 1.

                un = p.ultrasp[getIndex2D(1,l,lmax+1)]
                unm1 = 1.

                for n in range(1, nmax):
                    i1 = getIndex2D(n+1,l,lmax+1)
                    i2 = getIndex2D(n,l,lmax+1)
                    p.ultrasp[i1] = (p.c1[i2]*xi*un - p.c2[i2]*unm1) * p.c3[n]
                    unm1 = un
                    un = p.ultrasp[i1]
                    p.ultrasp1[i1] = ((p.twoalpha[l]+(n+1)-1.)*unm1-(n+1)*xi*p.ultrasp[i1]) / (p.twoalpha[l]*(1.-xi*xi))

            for m in range(0, lmax+1):
                i1 = getIndex2D(m,m,lmax+1)
                p.plm[i1] = 1.0
                if m > 0:
                    p.plm[i1] = (-1.)**m * p.dblfact[m] * sqrt(1.-costh*costh)**m

                plm1m = p.plm[i1]
                plm2m = 0.0

                for l in range(m+1, lmax+1):
                    i2 = getIndex2D(l,m,lmax+1)
                    p.plm[i2] = (costh*(2.*l-1.)*plm1m - (l+m-1.)*plm2m) / (l-m)
                    plm2m = plm1m
                    plm1m = p.plm[i2]

            p.dplm[0] = 0.

            for l in range(1, lmax+1):
                for m in range(0, l+1):
                    i1 = getIndex2D(l,m,lmax+1)
                    if l == m:
                        p.dplm[i1] = l*costh*p.plm[i1]/(costh*costh-1.0)
                    else:
                        i2 = getIndex2D(l-1,m,lmax+1)
                        p.dplm[i1] = (l*costh*p.plm[i1]-(l+m)*p.plm[i2]) / (costh*costh-1.0)

            # HACK: Cython doesn't support step in range
            count = (lmax-lmin+lskip) // lskip
            # for l in range(lmin, lmax+1, lskip):
            for l in range(count):
                l = lmin + lskip*l
                temp3 = 0.
                temp4 = 0.
                temp5 = 0.
                temp6 = 0.

                for m in range(0, l+1):
                    clm = 0.
                    dlm = 0.
                    elm = 0.
                    flm = 0.
                    for n in range(0, nmax+1):
                        i1 = getIndex2D(n,l,lmax+1)
                        i2 = getIndex3D(n,l,m,lmax+1,lmax+1)

                        clm = clm + p.ultrasp[i1]*p.cossum[i2]
                        dlm = dlm + p.ultrasp[i1]*p.sinsum[i2]
                        elm = elm + p.ultrasp1[i1]*p.cossum[i2]
                        flm = flm + p.ultrasp1[i1]*p.sinsum[i2]

                    i1 = getIndex2D(l,m,lmax+1)
                    temp3 = temp3 + p.plm[i1]*(clm*cosmphi[m]+dlm*sinmphi[m])
                    temp4 = temp4 - p.plm[i1]*(elm*cosmphi[m]+flm*sinmphi[m])
                    temp5 = temp5 - p.dplm[i1]*(clm*cosmphi[m]+dlm*sinmphi[m])
                    temp6 = temp6 - m*p.plm[i1]*(dlm*cosmphi[m]-clm*sinmphi[m])

                phinltil = r**l / (1.+r)**(2*l+1)
                b.Epot_bfe[k] = b.Epot_bfe[k] + temp3*phinltil
                ar = ar + phinltil * (-temp3*(l/r-(2.*l+1.)/(1.+r)) +
                                      temp4*4.*(2.*l+1.5)/(1.+r)**2)
                ath = ath + temp5*phinltil
                aphi = aphi + temp6*phinltil

            cosp = cos(phi)
            sinp = sin(phi)

            sinth = sqrt(1.-costh*costh)
            ath = -sinth*ath/r
            aphi = aphi/(r*sinth)

            b.ax[k] = config.G*(sinth*cosp*ar + costh*cosp*ath - sinp*aphi)
            b.ay[k] = config.G*(sinth*sinp*ar + costh*sinp*ath + cosp*aphi)
            b.az[k] = config.G*(costh*ar - sinth*ath)
            b.Epot_bfe[k] = b.Epot_bfe[k]*config.G

            # Update kinetic energy
            b.Ekin[k] = 0.5 * ((b.vx[k]-f.vx)**2 +
                               (b.vy[k]-f.vy)**2 +
                               (b.vz[k]-f.vz)**2)

cdef void external_field(Config config, Bodies b, COMFrame *f,
                         CPotential *pot, double strength, double *tnow):
    """
    Compute the acceleration, potential of the external tidal field.
    """
    cdef:
        int j, k
        double grad[3]
        double q[3]

    for k in range(config.n_bodies):
        q[0] = b.x[k]
        q[1] = b.y[k]
        q[2] = b.z[k]

        # Compute external potential gradient
        c_gradient(pot, tnow[0], &q[0], &grad[0])
        b.Epot_ext[k] = c_potential(pot, tnow[0], &q[0])

        b.ax[k] = b.ax[k] - strength*grad[0]
        b.ay[k] = b.ay[k] - strength*grad[1]
        b.az[k] = b.az[k] - strength*grad[2]

cdef void update_acceleration(Config config, Bodies b, Placeholders p,
                              COMFrame *f, CPotential *pot,
                              double extern_strength, double *tnow,
                              int *firstc):
    """
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
        calculation or not. If so, will call `internal_bfe_init()` to initialize
        the BFE coefficient / placeholder arrays.
    """
    cdef int j,k

    if config.selfgravitating:
        internal_bfe_field(config, b, p, f, firstc)
        external_field(config, b, f, pot, extern_strength, tnow)

    else:
        for k in range(config.n_bodies):
            b.ax[k] = 0.
            b.ay[k] = 0.
            b.az[k] = 0.
            b.Epot_ext[k] = 0.
            b.Epot_bfe[k] = 0.

        external_field(config, b, f, pot, extern_strength, tnow)
