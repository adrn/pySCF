extern int getIndex2D(int row, int col, int ncol);

extern int getIndex3D(int row, int col, int dep, int ncol, int ndep);

extern void accp_firstc(int nmax, int lmax, int zeroodd, int zeroeven,
                        double *dblfact,
                        double *twoalpha,
                        double *anltilde,
                        double *coeflm,
                        int *lmin, int *lskip,
                        double *c1,
                        double *c2,
                        double *c3);

extern void accp_LH(int nbodies, double *xyz, double *mass, int *ibound,
                    int nmax, int lmax, int zeroodd, int zeroeven,
                    double *sinsum, double *cossum,
                    double G, int *firstc,
                    double *dblfact, double *twoalpha, double *anltilde, double *coeflm,
                    int *lmin, int *lskip,
                    double *c1, double *c2, double *c3,
                    double *pot,
                    double *acc);

extern void acc_pot(int selfgrav, int nbodies, double *xyz, double *mass, int *ibound,
                    int nmax, int lmax, int zeroodd, int zeroeven,
                    double *sinsum, double *cossum,
                    double G, int *firstc,
                    double *dblfact, double *twoalpha, double *anltilde, double *coeflm,
                    int *lmin, int *lskip,
                    double *c1, double *c2, double *c3,
                    double *pot,
                    double *acc);

extern void frame(int iter, int n_recenter,
                  int nbodies, double *xyz, double *vxyz, double *mass,
                  double *pot);
