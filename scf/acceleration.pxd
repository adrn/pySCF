from gala.potential.potential.cpotential cimport CPotential
from structs cimport Config, Placeholders, Bodies, COMFrame

cdef void internal_bfe_init(Config config, Placeholders p)

cdef void internal_bfe_field(Config config, Bodies b, Placeholders p,
                             COMFrame *f, int *firstc)

cdef void external_field(Config config, Bodies b, COMFrame *f,
                         CPotential *pot, double strength, double *tnow)

cdef void update_acceleration(Config config, Bodies b, Placeholders p,
                              COMFrame *f, CPotential *pot,
                              double extern_strength, double *tnow,
                              int *firstc)
