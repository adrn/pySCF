typedef struct {
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

extern void accp_firstc(Config config, Placeholders p);

extern void accp_LH(Config config, double *xyz, double *mass, int *ibound,
                    Placeholders p, int *firstc,
                    double *pot, double *acc);

extern void acc_pot(Config config, double extern_strength,
                    double *xyz, double *mass, int *ibound,
                    Placeholders p, int *firstc,
                    double *pot, double *acc);

extern void frame(Config config, int iter,
                  double *xyz, double *vxyz, double *mass, double *pot,
                  int *pot_idx, double *xyz_frame, double *vxyz_frame);

extern void initvel(Config config, double *tnow, double *tvel, double dt,
                    double *vxyz, double *acc);
