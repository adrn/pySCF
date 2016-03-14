typedef struct {
    int n_bodies;
    int n_recenter;
    int n_snapshot;
    int n_tidal;
    int nmax;
    int lmax;
    int zeroodd;
    int zeroeven;
} Config;

extern void accp_firstc(Config config,
                        double *dblfact,
                        double *twoalpha,
                        double *anltilde,
                        double *coeflm,
                        int *lmin, int *lskip,
                        double *c1,
                        double *c2,
                        double *c3);

extern void accp_LH(Config config, double *xyz, double *mass, int *ibound,
                    double *sinsum, double *cossum,
                    double G, int *firstc,
                    double *dblfact, double *twoalpha, double *anltilde, double *coeflm,
                    int *lmin, int *lskip,
                    double *c1, double *c2, double *c3,
                    double *pot,
                    double *acc);

extern void acc_pot(Config config, int selfgrav,
                    double *xyz, double *mass, int *ibound,
                    double *sinsum, double *cossum,
                    double G, int *firstc,
                    double *dblfact, double *twoalpha, double *anltilde, double *coeflm,
                    int *lmin, int *lskip,
                    double *c1, double *c2, double *c3,
                    double *pot,
                    double *acc);

extern void frame(Config config, int iter,
                  double *xyz, double *vxyz, double *mass, double *pot,
                  int *pot_idx, double *xyz_frame);
