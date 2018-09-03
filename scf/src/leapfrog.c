#include "structs.h"

void step_pos(Config config, Bodies b, double dt,
              double *tnow, double *tpos) {
    int k;
    for (k=0; k<config.n_bodies; k++) {
        b.x[k] = b.x[k] + dt*b.vx[k];
        b.y[k] = b.y[k] + dt*b.vy[k];
        b.z[k] = b.z[k] + dt*b.vz[k];
    }
    *tpos = *tpos + dt;
    *tnow = *tpos;
}

void step_vel(Config config, Bodies b, double dt,
             double *tnow, double *tvel) {
    int k;
    for (k=0; k<config.n_bodies; k++) {
        b.vx[k] = b.vx[k] + dt*b.ax[k];
        b.vy[k] = b.vy[k] + dt*b.ay[k];
        b.vz[k] = b.vz[k] + dt*b.az[k];
    }
    *tvel = *tvel + dt;
    *tnow = *tvel;
}
