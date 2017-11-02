from gala.potential.potential.cpotential cimport CPotential
from structs cimport Config, Placeholders, Bodies, COMFrame

cdef void recenter_frame(Config config, Bodies b, COMFrame *f)

cdef void check_progenitor(int iter, Config config, Bodies b, Placeholders p,
                      COMFrame *f, CPotential *pot, double *tnow)

cdef void tidal_start(int iter, Config config, Bodies b, Placeholders p,
                      COMFrame *f, CPotential *pot, double *tnow, double *tpos,
                      double *tvel)
