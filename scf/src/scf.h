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
    double *pot;
    double *kin;
    double *mass;
    int *ibound;
    double *tub;
} Bodies;

extern void accp_firstc(Config config, Placeholders p);

extern void acc_pot(Config config, Bodies b, Placeholders p,
                    double extern_strength, int *firstc, double *xyz_frame);

extern void frame(int iter, Config config, Bodies b,
                  int *pot_idx, double *xyz_frame, double *vxyz_frame);

extern void step_vel(Config config, Bodies b, double dt,
                     double *tnow, double *tvel);

extern void tidal_start(Config config, Bodies b, Placeholders p,
                        double *tnow, double *tpos, double *tvel,
                        int *pot_idx, double *xyz_frame, double *vxyz_frame);

extern void step_system(int iter, Config config, Bodies b, Placeholders p,
                        double *tnow, double *tpos, double *tvel,
                        int *pot_idx, double *xyz_frame, double *vxyz_frame);
