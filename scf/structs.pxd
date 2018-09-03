cdef extern from "potential/src/cpotential.h" nogil:
    ctypedef struct CPotential:
        pass

cdef extern from "src/structs.h" nogil:
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

cdef extern from "src/leapfrog.c" nogil:
    void step_pos(Config config, Bodies b, double dt,
                  double *tnow, double *tvel) nogil

    void step_vel(Config config, Bodies b, double dt,
                  double *tnow, double *tvel) nogil
