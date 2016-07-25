from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
from math import factorial

# Third-party
from astropy.constants import G
from astropy import log as logger
from astropy.utils.data import get_pkg_data_filename
from astropy.io import ascii
import astropy.units as u
import numpy as np
import six

import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic, UnitSystem

# Project
from .helpers import (_test_accp_firstc, _test_accp_bfe, _test_tidal_start)

def test_accp_firstc():
    atol = 1E-8
    rtol = 1E-8
    split_pattr = "# -----------------------------------------------------------------------------"

    # don't change these
    nmax = 6
    lmax = 4

    # from Cython implementation
    d = _test_accp_firstc(nmax, lmax)

    # load "truth" from Fortran
    filename = get_pkg_data_filename('data/accp_firstc.txt')

    with open(filename,'r') as f:
        chunks = f.read().split(split_pattr)

    for chunk in chunks:
        tbl = ascii.read(chunk)
        name = tbl.colnames[-1]

        if len(tbl.colnames) == 2:
            for i,val in tbl:
                print(name, i)
                assert np.allclose(d[name][i], val, atol=atol, rtol=rtol)

        elif len(tbl.colnames) == 3:
            for i,j,val in tbl:
                print(name, i, j)
                assert np.allclose(d[name][i,j], val, atol=atol, rtol=rtol)

def test_accp_bfe():
    atol = 1E-7
    rtol = 1E-7
    skip = 1
    names = ['m','x','y','z','vx','vy','vz']
    bodies_filename = get_pkg_data_filename('data/SCFBI')
    bodies = np.genfromtxt(bodies_filename, dtype=None, names=names,
                           skip_header=skip)

    b = gd.CartesianPhaseSpacePosition(pos=bodies[['x','y','z']].view(np.float64).reshape(-1,3).T,
                                       vel=bodies[['vx','vy','vz']].view(np.float64).reshape(-1,3).T)

    d = _test_accp_bfe(b)

    # plm - load "truth" from Fortran
    f77_tbl = ascii.read(get_pkg_data_filename('data/plm.txt'))
    for l,m,plm in f77_tbl:
        assert np.allclose(plm,d['plm'][l,m], atol=atol, rtol=rtol)

    # sinsum and cossum - load "truth" from Fortran
    f77_tbl = ascii.read(get_pkg_data_filename('data/sincossum.txt'))
    for n,l,m,sinsum,cossum in f77_tbl:
        assert np.allclose(sinsum, d['sinsum'][n,l,m], atol=atol, rtol=rtol)
        assert np.allclose(cossum, d['cossum'][n,l,m], atol=atol, rtol=rtol)

    # ultrasp and ultraspt - load "truth" from Fortran
    f77_tbl = ascii.read(get_pkg_data_filename('data/ultrasp.txt'))
    for n,l,ultrasp,ultraspt in f77_tbl:
        assert np.allclose(ultrasp, d['ultrasp'][n,l], atol=atol, rtol=rtol)
        assert np.allclose(ultraspt, d['ultraspt'][n,l], atol=atol, rtol=rtol)

    # acceleration and potential at position of bodies
    f77_tbl = ascii.read(get_pkg_data_filename('data/accp_bfe.txt'))
    for n,f77_row in enumerate(f77_tbl):
        assert np.allclose(np.array(list(f77_row)),
                           [d['ax'][n], d['ay'][n], d['az'][n], d['pot'][n]], atol=atol, rtol=rtol)

def test_tidal_start():
    atol = 1E-7
    rtol = 1E-7

    skip = 1
    names = ['m','x','y','z','vx','vy','vz']
    bodies_filename = get_pkg_data_filename('data/SCFBI')
    bodies = np.genfromtxt(bodies_filename, dtype=None, names=names,
                           skip_header=skip)
    b = gd.CartesianPhaseSpacePosition(pos=bodies[['x','y','z']].view(np.float64).reshape(-1,3).T,
                                       vel=bodies[['vx','vy','vz']].view(np.float64).reshape(-1,3).T)

    # Matched to scf/fortran/SCFPAR
    rs = 10.*u.kpc
    M = ((220.*u.km/u.s)**2 * rs / G).to(u.Msun)
    potential = gp.HernquistPotential(m=M, c=rs, units=galactic)
    w0 = gd.CartesianPhaseSpacePosition(pos=[10.,0,0]*u.kpc,
                                        vel=[0,75.,0]*u.km/u.s)

    # define unit system for simulation
    mass_scale = 2.5e4*u.Msun
    length_scale = 0.01*u.kpc
    l_unit = u.Unit('{} kpc'.format(length_scale.to(u.kpc).value))
    m_unit = u.Unit('{} Msun'.format(mass_scale.to(u.Msun).value))
    _G = G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value
    t_unit = u.Unit("{:08f} Myr".format(np.sqrt(l_unit.scale**3 / (_G*m_unit.scale))))
    a_unit = u.radian
    units = UnitSystem(l_unit, m_unit, t_unit, a_unit)

    # transform potential to simulation units
    Potential = potential.__class__
    _potential = Potential(units=units, **potential.parameters)

    d = _test_tidal_start(_potential.c_instance, w0, b, n_tidal=128,
                          length_scale=length_scale, mass_scale=mass_scale)

    # compare velocities after stepping by half step
    f77_tbl = ascii.read(get_pkg_data_filename('data/vel_after_init.txt'))
    for n,f77_row in enumerate(f77_tbl):
        assert np.allclose(np.array(list(f77_row)), d['v_after_init'].T[n], atol=atol, rtol=rtol)

    # compare positions and velocities after stepping one whole step
    f77_tbl = ascii.read(get_pkg_data_filename('data/posvel_one_step.txt'))
    for n,f77_row in enumerate(f77_tbl):
        x_v = np.array(list(f77_row))
        xyz = x_v[:3]
        vxyz = x_v[3:]
        assert np.allclose(xyz, d['xyz_one_step'].T[n], atol=atol, rtol=rtol)
        assert np.allclose(vxyz, d['vxyz_one_step'].T[n], atol=atol, rtol=rtol)

    # compare positions and velocities at end of tidal start
    f77_tbl = ascii.read(get_pkg_data_filename('data/posvel_tidal_start_end.txt'))
    for n,f77_row in enumerate(f77_tbl):
        x_v = np.array(list(f77_row))
        xyz = x_v[:3]
        vxyz = x_v[3:]
        assert np.allclose(xyz, d['xyz_end'].T[n], atol=atol, rtol=rtol)
        assert np.allclose(vxyz, d['vxyz_end'].T[n], atol=atol, rtol=rtol)

    print(d['m_prog'])
    print(d['frame_xyz'])
    print(d['frame_vxyz'])

def test_against_biff():
    import biff

    # don't change! these are set in _test_accp_bfe as well
    nmax = 6
    lmax = 4

    skip = 1
    names = ['m','x','y','z','vx','vy','vz']
    bodies_filename = get_pkg_data_filename('data/SCFBI')
    bodies = np.genfromtxt(bodies_filename, dtype=None, names=names,
                           skip_header=skip)
    b = gd.CartesianPhaseSpacePosition(pos=bodies[['x','y','z']].view(np.float64).reshape(-1,3).T,
                                       vel=bodies[['vx','vy','vz']].view(np.float64).reshape(-1,3).T)

    # coeffs computed with SCF code
    res = _test_accp_bfe(b)
    scf_acc = np.vstack((res['ax'], res['ay'], res['az'])).T

    # coeffs computed with Biff
    mass = np.ones(b.pos.shape[1]) / b.pos.shape[1]
    biff_xyz = np.ascontiguousarray(b.pos.value.T)
    S,T = biff.compute_coeffs_discrete(biff_xyz, mass, nmax, lmax, 1.,
                                       skip_odd=False, skip_even=False, skip_m=False)
    biff_acc = -biff.gradient(biff_xyz, S, T, G=1., M=1., r_s=1.)

    for n in range(nmax+1):
        for l in range(lmax+1):
            for m in range(l+1):
                # transform from H&O 1992 coefficients to Lowing 2011 coefficients
                if l != 0:
                    fac = np.sqrt(4*np.pi) * np.sqrt((2*l+1) / (4*np.pi) * factorial(l-m) / factorial(l+m))
                    res['cossum'][n,l,m] /= fac
                    res['sinsum'][n,l,m] /= fac

    frac_diff = (-np.array(res['cossum']) - S) / S
    assert np.allclose(frac_diff[np.isfinite(frac_diff)], 0.)
    assert np.allclose(biff_acc, scf_acc)
