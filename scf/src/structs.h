#ifndef _STRUCT_DEFS
#define _STRUCT_DEFS
    typedef struct {
        int n_steps;
        double dt;
        double t0;
        int n_bodies;
        int n_recenter;
        int n_snapshot;
        int n_tidal;
        int nmax;
        int lmax;
        int zeroodd;
        int zeroeven;
        int selfgravitating;
        double ru;
        double mu;
        double vu;
        double tu;
        double G;
    } Config;

    typedef struct {
        double *dblfact;
        double *twoalpha;
        double *anltilde;
        double *coeflm;
        double *plm;
        double *dplm;
        double *ultrasp;
        double *ultraspt;
        double *ultrasp1;
        double *sinsum;
        double *cossum;
        double *c1;
        double *c2;
        double *c3;
        int *lmin;
        int *lskip;
        double *pot0;
        double *kin0;
        double *ax0;
        double *ay0;
        double *az0;
    } Placeholders;

    typedef struct {
        double *x;
        double *y;
        double *z;
        double *vx;
        double *vy;
        double *vz;
        double *ax;
        double *ay;
        double *az;
        double *Epot_ext;
        double *Epot_bfe;
        double *Ekin;
        double *mass;
        int *ibound;
        double *tub;
    } Bodies;

    typedef struct {
        double m_prog;
        double x;
        double y;
        double z;
        double vx;
        double vy;
        double vz;
        int *pot_idx; // used for sorting particles on how bound they are
    } COMFrame; // Center-of-mass reference frame
#endif
