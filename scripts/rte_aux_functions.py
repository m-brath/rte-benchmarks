#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 11:24:07 2026

@author: u242031
"""

import re
import numpy as np
import periodictable as pt
from scipy import constants as scc
import pyarts as pa




# =============================================================================
# %% paths/constants
# =============================================================================

#universal gas constant
R=scc.R

#gravity
g=scc.g  # m s^{-2}

#Avogadro constant
N_A=scc.Avogadro

#atomic mass
m_a=scc.atomic_mass

# Radius of Earth
R_e = 6371229  # m

# Molar mass of dry air# Donald P. Gatley, Sebastian Herrmann & Hans-Joachim Kretzschmar
# (2008) A Twenty-First Century Molar Mass for Dry Air, HVAC&R Research, 14:5, 655-662, DOI:
# 10.1080/10789669.2008.10391032
M_d = 28.965369e-3  # kg mol^-1

R_D = R / M_d  # J K^-1 kg^-1



# =============================================================================
# Helper: molecular formula formatter
# =============================================================================



def molecular_mass(formula: str) -> float:
    """Calculate the molar mass of a molecule from its molecular formula.

    The formula must use correct capitalisation (e.g. 'H2O', 'CO2', 'HCl').
    Use :func:`format_molecular_formula` first if the input case is uncertain.

    Parameters
    ----------
    formula : str
        Properly capitalised molecular formula (e.g. 'CH3Cl', 'HCl', 'N2O').

    Returns
    -------
    float
        Molar mass in g mol⁻¹.

    Examples
    --------
    >>> molecular_mass('H2O')   # 18.015
    >>> molecular_mass('CO2')   # 44.010
    >>> molecular_mass('HCl')   # 36.461
    """
    mass = 0.0
    for symbol, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula):        
        if not symbol:
            continue
        n = int(count) if count else 1
        mass += pt.elements.symbol(symbol).mass * n
    return mass*N_A*m_a, mass


def get_species_masses(species: list[str]) -> dict[str, float]:
    """Return a dict mapping each species formula to its molar mass in g mol⁻¹.

    Parameters
    ----------
    species : list[str]
        List of properly capitalised molecular formula strings, e.g. as
        returned by :func:`format_species_list`.

    Returns
    -------
    dict[str, float]
        ``{formula: molar_mass_g_per_mol}`` for every entry in *species*.

    Examples
    --------
    >>> get_species_masses(['H2O', 'CO2', 'O3'])
    {'H2O': 18.015, 'CO2': 44.010, 'O3': 47.998}
    """

    output={}
    for s in species:

        # hardcode some special cases for CFC-11 and CFC-12
        if s == 'CFC11':
            output[s], _ = molecular_mass('CCl3F')
        elif s == 'CFC12':
            output[s], _ = molecular_mass('CCl2F2')
        else:
            output[s], _ = molecular_mass(s)

    return output


# =============================================================================
# %% aux
# =============================================================================

def convert_time(data, obs_type):
    """
    Convert time variables from a dataset to an array of numpy datetime64 objects.
    This function extracts time and date information from a dataset containing
    observation data in either 'arsa' or 'iasi' format, and converts them into
    numpy datetime64 objects.
    Parameters
    ----------
    data : xarray.Dataset or dict-like
        The dataset containing the time variables. It must include variables
        named 'hhmmss{obs_type}' and 'yyyymmdd{obs_type}', where {obs_type}
        is either 'arsa' or 'iasi'.
        - 'hhmmss{obs_type}': Time in HHMMSS format (e.g., 123456 for 12:34:56).
        - 'yyyymmdd{obs_type}': Date in YYYYMMDD format (e.g., 230101 for 2023-01-01).
          Note: The year is stored as a 2-digit value and 2000 is added to it.
    obs_type : str
        The type of observation data. Must be either 'arsa' or 'iasi'.
    Returns
    -------
    numpy.ndarray
        An array of numpy datetime64 objects representing the converted timestamps
        in the format 'YYYY-MM-DDTHH:MM:SS'.
    Raises
    ------
    ValueError
        If `obs_type` is not 'arsa' or 'iasi'.
    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> data = xr.Dataset({
    ...     'hhmmssarsa': ('time', [123456, 235959]),
    ...     'yyyymmddarsa': ('time', [230101, 231231])
    ... })
    >>> convert_time(data, 'arsa')
    array(['2023-01-01T12:34:56', '2023-12-31T23:59:59'], dtype='datetime64[s]')
    Notes
    -----
    There is a variable shadowing issue in the function: `mm` is first assigned
    from the time variable ('hhmmss') but is then overwritten by the month value
    from the date variable ('yyyymmdd'). As a result, the minutes in the output
    datetime will always reflect the month value, not the actual minutes.
    """


    if not obs_type in ['arsa', 'iasi']:
        raise ValueError("obs_type must be either 'arsa' or 'iasi'")
        
    var_name1 = "hhmmss" + obs_type

    hhmmss = data[var_name1].values.astype(int)
    hh = hhmmss // 10000
    mm = (hhmmss % 10000) // 100
    ss = hhmmss % 100


    var_name2 = "yyyymmdd" + obs_type

    yyyymmdd = data[var_name2].values.astype(int)
    yyyy = yyyymmdd // 10000 +2000
    mm = (yyyymmdd % 10000) // 100
    dd = yyyymmdd % 100

    # put them together to a datetime object
    time = np.array(
        [
            np.datetime64(f"{y:04d}-{m:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}")
            for y, m, d, h, mi, s in zip(yyyy, mm, dd, hh, mm, ss)
        ]
    )

    return time

# =============================================================================
# %% atmospheric 
# =============================================================================

M_W, _ = molecular_mass("H2O")
R_W = R / M_W


def T_virtual(T,r_v):
    """
    Calculate the virtual temperature.
    The virtual temperature is the temperature that dry air would need to have 
    to match the density of a sample of moist air at the same pressure.
    Parameters
    ----------
    T : float or array-like
        Actual temperature [K].
    r_v : float or array-like
        Water vapor mixing ratio [kg/kg].
    Returns
    -------
    Tv : float or array-like
        Virtual temperature [K].
    Notes
    -----
    The virtual temperature is calculated using the formula:
        Tv = T * (1 + r_v / epsilon) / (1 + r_v)
    where epsilon = R_D / R_W is the ratio of the gas constant for dry air (R_D) 
    to the gas constant for water vapor (R_W), with typical values of 
    R_D = 287.05 J/(kg·K) and R_W = 461.51 J/(kg·K), giving epsilon ≈ 0.622.
    
    """

    
    epsilon=R_D/R_W
    
    Tv=T*(1+r_v/epsilon)/(1+r_v)
    
    return Tv

def mixingratio2vmr(r,Mx):
    """
    Convert mixing ratio to volume mixing ratio (VMR).
    Parameters
    ----------
    r : float or array-like
        Mixing ratio of the gas [kg/kg].
    Mx : float
        Molar mass of the gas [kg/mol].
    Returns
    -------
    float or array-like
        Volume mixing ratio (VMR) of the gas [mol/mol].
    Notes
    -----
    The conversion uses the molar mass of dry air (M_d) defined as a global
    constant in the module. The formula used is:
        VMR = r / (r + Mx / Md)
    where Md is the molar mass of dry air [kg/mol].
    Examples
    --------
    >>> # Convert water vapor mixing ratio to VMR
    >>> r = 0.01  # kg/kg
    >>> Mx = 18.015  # g/mol (water vapor)
    >>> vmr = mixingratio2vmr(r, Mx)
    """
   
    Md=M_d
    
    return r/(r+Mx/Md)


def get_altitude(p, T, r_v,z0=0):
    """
    Calculate altitude at each model level using the hypsometric equation.
    This function integrates the hypsometric equation from the surface upwards
    to compute the geometric height at each model level, accounting for water
    vapor through the virtual temperature.
    Parameters
    ----------
    p : np.ndarray
        Pressure at each model level, shape ( n_levels) [Pa].
        The array is expected to start at the surface (SFC).
    T : np.ndarray
        Temperature at each model level, shape (n_levels) [K].
        The array is expected to start at the SFC.
    r_v : np.ndarray
        Water vapor mixing ratio at each model level,
        shape (n_levels) [kg/kg].
    z0 : float, optional
        Surface altitude (starting height for integration) [m].
        Default is 0.
    Returns
    -------
    Z : np.ndarray
        Altitude at each model level, shape (n_levels) [m].
    Notes
    -----
    The altitude is computed using the hypsometric equation:
        dz = (R_D / g) * T_v * d(ln(p))
    where:
        - R_D is the specific gas constant for dry air [J/(kg·K)]
        - g is the gravitational acceleration [m/s²]
        - T_v is the virtual temperature [K], accounting for water vapor
        - d(ln(p)) is the natural log of the pressure ratio between adjacent levels
    The integration starts at the surface (first index) and proceeds upward
    toward the TOA (last index). 
    
    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([101325, 85000, 70000])  # Pressure in Pa
    >>> T = np.array([288.15, 278.15, 268.15])  # Temperature in K
    >>> r_v = np.zeros_like(T)  # No water vapor
    >>> Z = get_altitude(p, T, r_v, z0=0)
    """
    
    # allocate
    Z = np.zeros_like(T)
    
    # set the surface altitude
    Z[0] = z0

    # model levels
    ml = np.arange(0, np.size(T, 0))

    # set the initial height to the surface altitude
    z_h =Z[0]
    
    # have to start the integration at the surface
    for level in ml[1:]:

        # compute virtual temperature
        t_level = T_virtual(T[level], r_v[level])

        # compute the logarithmic pressure difference between the current and previous level
        dlog_p = np.log(p[level-1] / p[level])
        
        # integrate the hypsometric equation to get the height at the current level
        z_h = z_h + (t_level * dlog_p)*R_D/g


        Z[level] = z_h


    return Z


def lin_interp(x, xp, fp):
    """Linearly interpolate values with optional linear extrapolation.

    This helper wraps :func:`numpy.interp` and ensures the independent grid
    points are sorted before interpolation. It also filters out NaN entries in
    the interpolation grid and defines linear extrapolation at the boundaries
    based on the first and last two values of *fp*.

    Parameters
    ----------
    x : array-like
        Query points where the interpolated values are computed.
    xp : array-like
        One-dimensional array of independent variable values. Must contain at
        least two points.
    fp : array-like
        One-dimensional array of dependent variable values, with the same
        shape as *xp*.

    Returns
    -------
    numpy.ndarray
        Interpolated (and extrapolated) values at the points in *x*.

    Notes
    -----
    - If *xp* is not strictly increasing, it is sorted along with *fp*.
    - NaN values in *xp* or *fp* are removed before interpolation.
    - Linear extrapolation is applied outside the *xp* range using boundary
      slopes derived from the first and last two valid points.

    Examples
    --------
    >>> x = np.array([0.5, 1.5])
    >>> xp = np.array([0.0, 1.0, 2.0])
    >>> fp = np.array([0.0, 1.0, 4.0])
    >>> lin_interp(x, xp, fp)
    array([0.5, 2.5])
    """

    # remove NaN values from xp and fp
    mask = ~np.isnan(xp) & ~np.isnan(fp)
    xp = xp[mask]
    fp = fp[mask]


    # make sure that xp is in ascending order
    if not np.all(np.diff(xp) > 0):
        
        # sort xp and fp in ascending order of xp
        sort_idx = np.argsort(xp)
        xp = xp[sort_idx]
        fp = fp[sort_idx]

    # check if xp has at least 2 points
    if len(xp) < 2:
        raise ValueError("xp must have at least 2 points for interpolation.")

    # make sure that x is in ascending order
    if not np.all(np.diff(x) > 0):
        # sort x in ascending order
        x = np.sort(x)    
    
    left=(3*fp[0]-fp[1])/2
    right=(3*fp[-1]-fp[-2])/2
    
    return np.interp(x, xp, fp, left=left, right=right)


def make_gridded_field(data_temp, species_in_data, p_grid, arts_names=[]):
    """Build an ARTS GriddedField4 from temperature, species, and pressure data.

    This helper constructs a pyarts ``GriddedField4`` object containing
    temperature, altitude, and absorption species fields. The function
    ensures the pressure grid is ordered from high to low pressure, fills the
    ARTS grid with profile values, and computes the altitude using the
    temperature and water vapor mixing ratio.

    Parameters
    ----------
    data_temp : numpy.ndarray
        Profile data array with shape ``(n_levels, n_variables)``. The first
        column is expected to contain temperature values, and the water vapor
        mixing ratio column is located at ``species_in_data.index('h2o') + 2``.
    species_in_data : list[str]
        Ordered list of species identifiers corresponding to the absorption
        species columns in *data_temp*.
    p_grid : array-like
        Pressure grid values for the second dimension of the ARTS field.
        Pressures are expected in descending order; if not, the function
        reverses the grid and the associated profile data.

    Returns
    -------
    pyarts.arts.GriddedField4
        ARTS gridded field containing the ``T``, ``z``, and absorption species
        dimensions.

    Notes
    -----
    - Dimension 0 is set to ``['T', 'z'] + abs_species``.
    - The altitude field is computed via :func:`get_altitude`.
    - A simple surface altitude estimate ``z0`` is derived from the top
      pressure value.
    """

    #check if p_grid is in descending order, if not , reverse it
    if p_grid[0] < p_grid[-1]:
        p_grid = p_grid[::-1]
        data_temp = data_temp[::-1, :]

 

    # Create a GriddedField4 object
    atm_field = pa.arts.GriddedField4()
    
    # set up grids
    if len(arts_names)>0:
        abs_species = [f"abs_species-{name}" for name in arts_names]
    else:
        abs_species = [f"abs_species-{key}" for key in species_in_data]

    atm_field.set_grid(0, ["T", "z"] + abs_species)
    atm_field.set_grid(1, p_grid)
    atm_field.data = np.zeros((len(atm_field.grids[0]), len(atm_field.grids[1]), 1, 1))    
    atm_field.data[:,:,0,0]=data_temp.T

    #find the index of the the H2O column in the data_temp array
    h2o_index = species_in_data.index('h2o') + 2

    #add altitude 
    z0=16e3 * (5 - np.log10(p_grid[0]))
    if z0<0:
        z0=0
    atm_field[1,:,0,0]=get_altitude(p_grid, data_temp[:, 0], data_temp[:, h2o_index], z0)
    
    return atm_field