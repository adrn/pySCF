from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import numpy as np
import h5py

# Project
from .scf import _scf

class SCFSimulation(object):
    """

    Parameters
    ----------
    bodies : :class:`gary.dynamics.CartesianPhaseSpacePosition`
        Dimensionless position and velocity of the satellite mass
        particles (the 'N bodies').
    potential : :class:`gary.potential.CPotentialBase`
        The external potential (e.g., host galaxy) as an instance of
        a subclass of the `CPotentialBase` class.
    mass_scale : :class:`astropy.units.Quantity`
        TODO:
    length_scale :class:`astropy.units.Quantity`
        TODO:
    self_gravity : bool (optional)
        Option to turn off the satellite self-gravity (default = True).
    nmax : int (optional)
        Number of radial eigenfunctions used in the basis function expansion
        of the satellite potential (default = 6).
    lmax : int (optional)
        Number of angular eigenfunctions used in the basis function expansion
        of the satellite potential (default = 4).
    zero_odd : bool (optional)
        Set all odd terms in the basis function expansion of the satellite
        potential (default = False).
    zero_even : bool (optional)
        Set all even terms in the basis function expansion of the satellite
        potential (default = False).
    """
    def __init__(self, bodies, potential, mass_scale, length_scale,
                 self_gravity=True, nmax=6, lmax=4, zero_odd=False, zero_even=False,
                 output_path=None, snapshot_filename="snap.h5"):

        self.mass_scale = u.Quantity(mass_scale)
        self.length_scale = u.Quantity(length_scale)

        if output_path is None:
            output_path = os.getcwd()

        pass

    def run(self, w0, dt, n_steps, t0=0.,
            n_snapshot=None, n_recenter=256, n_tidal=256):
        """
        Run the N-body simulation.

        Parameters
        ----------

        """

