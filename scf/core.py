from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy.constants import G
import astropy.units as u
import numpy as np

import gala.dynamics as gd
import gala.potential as gp
from gala.units import UnitSystem

# Project
from .scf import run_scf

__all__ = ['SCFSimulation']

class SCFSimulation(object):
    """

    Parameters
    ----------
    bodies : :class:`gala.dynamics.CartesianPhaseSpacePosition`
        Dimensionless position and velocity of the satellite mass
        particles (the 'N bodies').
    potential : :class:`gala.potential.CPotentialBase`
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

    @u.quantity_input(mass_scale=u.Msun, length_scale=u.kpc)
    def __init__(self, bodies, potential, mass_scale, length_scale,
                 self_gravity=True, nmax=6, lmax=4, zero_odd=False, zero_even=False,
                 output_path=None):

        if not isinstance(bodies, gd.CartesianPhaseSpacePosition):
            raise ValueError("bodies must be a CartesianPhaseSpacePosition instance.")
        self.bodies = bodies

        if not isinstance(potential, gp.CPotentialBase):
            raise ValueError("Potential object must be an instance of a "
                             "CPotentialBase subclass.")

        self.mass_scale = mass_scale
        self.length_scale = length_scale
        self.units = self.units_from_scales(self.mass_scale, self.length_scale)

        self.self_gravity = bool(self_gravity)
        self.nmax = int(nmax)
        self.lmax = int(lmax)
        self.zero_odd = bool(zero_odd)
        self.zero_even = bool(zero_even)

        if output_path is None:
            output_path = os.getcwd()

        self.output_path = os.path.abspath(output_path)

        # transform potential to simulation units
        self.potential = potential
        Potential = self.potential.__class__
        self._potential = Potential(units=self.units, **self.potential.parameters)

    # @u.quantity_input(dt=u.Myr)
    def run(self, w0, dt, n_steps, t0=0.,
            n_snapshot=None,  n_tidal=256,
            snapshot_filename="scfoutput.h5", overwrite=False):
        """
        Run the N-body simulation.

        Parameters
        ----------
        w0 : :class:`gala.dynamics.CartesianPhaseSpacePosition`
            Initial conditions for the satellite orbit.
        dt : :class:`astropy.units.Quantity`, float
            Timestep. If no unit is provided, assumed to be in simulation units.
        n_steps : int
            Number of integration steps to make.
        t0 : :class:`astropy.units.Quantity`, float (optional)
            Starting time of the simulation. If no unit is provided, assumed
            to be in simulation units.
        n_snapshot : int (optional)
            How often to output a snapshot. Set to `None` to only save the final
            phase-space positions of the bodies.
        n_tidal : int (optional)
            Number of steps to slowly turn on the external tidal field.
        snapshot_filename : str (optional)
            Name of the file to store simulation snapshots.
        overwrite : bool (optional)
            Overwrite existing file.

        TODO: n_recenter is set to 1 because I was having issues with
        energy conservation...

        """
        if not isinstance(w0, gd.CartesianPhaseSpacePosition):
            raise ValueError("Satellite initial conditions, w0, must be a "
                             "CartesianPhaseSpacePosition instance.")

        if n_snapshot is None:
            n_snapshot = 0

        n_recenter = 1
        if n_recenter <= 0:
            raise ValueError("n_recenter must be > 0")

        if n_tidal < 0:
            raise ValueError("n_tidal must be >= 0")

        output_file = os.path.join(self.output_path, snapshot_filename)
        if os.path.exists(output_file) and overwrite:
            os.remove(output_file)

        run_scf(self._potential.c_instance, w0, self.bodies,
                self.mass_scale, self.length_scale,
                dt, n_steps, t0,
                n_snapshot=n_snapshot, n_recenter=n_recenter, n_tidal=n_tidal,
                nmax=self.nmax, lmax=self.lmax,
                zero_odd=self.zero_odd, zero_even=self.zero_even,
                self_gravity=self.self_gravity, output_file=output_file)

    @staticmethod
    def units_from_scales(mass_scale, length_scale):
        # define unit system for simulation
        l_unit = u.Unit(length_scale.to(u.kpc))
        m_unit = u.Unit(mass_scale.to(u.Msun))
        t_unit = u.Unit(np.sqrt((l_unit**3) / (G*m_unit)).to(u.Myr))
        v_unit = u.Unit((l_unit / t_unit).to(u.km/u.s))
        a_unit = u.radian
        return UnitSystem(l_unit, m_unit, t_unit, a_unit, v_unit)
