from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import numpy as np

# Project
# ...

def movie(sim, snapshot_filename, units=None):
    """
    Make a movie of particle positions from each snapshot in the
    simulation.

    Parameters
    ----------
    sim : :class:`scf.SCFSimulation`
        An instance of the simulation class containing metadata
        about the run.
    snapshot_filename : str
        The path to the HDF5 snapshot file.
    units : :class:`gala.units.UnitSystem` (optional)
        A unit system to convert to before plotting. Default is to
        plot in simulation units (``units=None``).

    Returns
    -------
    anim : :class:`matplotlib.animation.FuncAnimation`
        A ``matplotlib`` animation instance.

    """
    pass
