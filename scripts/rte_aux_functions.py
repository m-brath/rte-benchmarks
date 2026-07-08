#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 11:24:07 2026

@author: u242031
"""


import os
import numpy as np
from scipy import constants as scc
import xarray as xr
import pyarts as pa
import FluxSimulator as fsm





# =============================================================================
# %% paths/constants
# =============================================================================

#universal gas constant
R=scc.R

#gravity
g=scc.g  # m s^{-2}


# Radius of Earth
R_e = 6371229  # m

# Molar mass of dry air# Donald P. Gatley, Sebastian Herrmann & Hans-Joachim Kretzschmar
# (2008) A Twenty-First Century Molar Mass for Dry Air, HVAC&R Research, 14:5, 655-662, DOI:
# 10.1080/10789669.2008.10391032
M_d = 28.965369e-3  # kg mol^-1

R_D = R / M_d  # J K^-1 kg^-1

# Molecular mass of water
#Value taken from the periodictable python packacge
#Kienzle, P. A. (2008). Extensible periodic table [Computer Software]
#https://doi.org/10.5281/zenodo.18809123
#https://github.com/python-periodictable/periodictable
M_W=0.01801500001894382

# water vapor gas constant
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

    #remove any negative values
    atm_field.data[atm_field.data<0]=0


    #find the index of the the H2O column in the data_temp array
    h2o_index = species_in_data.index('h2o') + 2

    #add altitude
    z0=16e3 * (5 - np.log10(p_grid[0]))
    if z0<0:
        z0=0
    atm_field[1,:,0,0]=get_altitude(p_grid, data_temp[:, 0], data_temp[:, h2o_index], z0)

    return atm_field

# =============================================================================
# pyarts/fluxsim functions
# =============================================================================

def get_Ncols_and_Nvariants(aux_in):
    """Extract number of columns and variants from auxiliary data.

    Parameters
    ----------
    aux_in : list
        List of auxiliary data objects containing grid information.

    Returns
    -------
    tuple
        (N_cols, N_variants, idx_col, idx_var) where:
        - N_cols : int - Number of columns
        - N_variants : int - Number of variants
        - idx_col : int - Index of column_index in grids
        - idx_var : int - Index of variant_index in grids
    """
    #find index of columns and variants
    grids=aux_in[0].grids[0]
    idx_col=[i for i, gr in enumerate(grids) if 'column_index' in str(gr)][0]
    idx_var=[i for i, gr in enumerate(grids) if 'variant_index' in str(gr)][0]

    cols=[aux[idx_col] for aux in aux_in]
    variants=[aux[idx_var] for aux in aux_in]

    N_cols=int(np.max(cols)+1)
    N_variants=int(np.max(variants)+1)

    return N_cols, N_variants, idx_col, idx_var


def define_abs_species(SW_flxsim, species_list_of_data):
    """Define absorption species for FluxSimulator based on available species.

    Parameters
    ----------
    SW_flxsim : FluxSimulator
        FluxSimulator object with available line and cross-section species.
    species_list_of_data : list
        List of species names to process.

    Returns
    -------
    list
        List of absorption species names with appropriate suffixes (e.g., '-XFIT').
    """
    #set abs_species
    abs_species=[]


    for i, spc in enumerate(species_list_of_data):

        temp=[spc_arts for spc_arts in SW_flxsim.line_species_available if spc_arts == spc]

        if len(temp)==1:
            abs_species.append(spc)

        temp=[spc_arts for spc_arts in SW_flxsim.xsec_species_available if spc_arts == spc]

        if len(temp)==1:
            abs_species.append(spc+'-XFIT')

    return abs_species


def set_sun_position(SW_flxsim, sza, tsi, atm):
        
    #Set sun to get sun parameter
    SW_flxsim.set_sun()
    sun_dist=SW_flxsim.get_sun()[0].distance*1.    
    
    #set sun position according to Sect 4.1 of star arts paper
    Toa_altitude = SW_flxsim.ws.refellipsoid.value[0]+atm[1,-1,0,0]
    phi=sza-np.rad2deg(np.arcsin(Toa_altitude/sun_dist*np.sin(np.pi-np.deg2rad(sza))))

    SW_flxsim.ws.sunsChangeGeometry(distance=sun_dist, latitude=0, longitude=phi, index=0)
    SW_flxsim.scale_sun_to_specific_TSI_at_TOA(tsi, 0, 0, atm[1,-1,0,0])

def export_to_xarray(Result, N_variants, N_cols, n_levels, n_freqs, f_grid, results_folder='', export_results=True):
        # Create xarray dataset
    ds = xr.Dataset(
        {
            'altitude': (['variant','column','level'], Result['altitude'], {'units': 'm'}),
            'pressure': (['variant','column','level'], Result['pressure'], {'units': 'Pa'}),
            'flux_clearsky_up': (['variant','column','level'], Result['flux_clearsky_up'], {'units': 'W/m^2'}),
            'flux_clearsky_down': (['variant','column','level'], Result['flux_clearsky_down'], {'units': 'W/m^2'}),
            'spectral_flux_up_TOA': (['variant','column','frequency'], Result['spectral_flux_up_TOA'], {'units': 'W/m^2/Hz'}),
            'spectral_flux_down_TOA': (['variant','column','frequency'], Result['spectral_flux_down_TOA'], {'units': 'W/m^2/Hz'}),
            'spectral_flux_down_SFC': (['variant','column','frequency'], Result['spectral_flux_down_SFC'], {'units': 'W/m^2/Hz'}),
            'spectral_flux_up_SFC': (['variant','column','frequency'], Result['spectral_flux_up_SFC'], {'units': 'W/m^2/Hz'}),
        },
        coords={
            'variant': np.arange(N_variants),
            'column': np.arange(N_cols),
            'level': np.arange(n_levels),
            'frequency': (['frequency'], f_grid, {'units': 'Hz'}),
        }
    )

    if export_results:
        if results_folder=='':
            results_folder=os.getcwd()
        os.makedirs(results_folder, exist_ok=True)
        ds.to_netcdf(os.path.join(results_folder, f'Reference_fluxes_Nf{n_freqs}.nc'))

    return ds



def rte_benchmark_sw(data_in, aux_in, f_grid, results_folder, setup_name, export_results=True, reverse_vertical_order=True):
    """Compute shortwave radiative transfer using FluxSimulator.

    Parameters
    ----------
    data_in : list
        List of atmospheric data objects.
    aux_in : list
        List of auxiliary data objects (surface properties, etc.).
    f_grid : array
        Frequency grid in Hz.
    results_folder : str
        Path to folder for saving results.
    setup_name : str
        Name of the setup/scenario.
    export_results : bool, optional
        Whether to export results to NetCDF file (default: True).
    reverse_vertical_order : bool, optional
        Whether to reverse vertical order of output arrays (default: True).
        This is the order of the RFMIP data, which is from top to bottom,
        while the output of the FluxSimulator is from bottom to top.

    Returns
    -------
    tuple
        (ds, SW_flxsim) where:
        - ds : xr.Dataset - xarray dataset with computed fluxes
        - SW_flxsim : FluxSimulator - Shortwave FluxSimulator object
    """
    # get number of columns and variants in the input data
    N_cols, N_variants, idx_col, idx_var = get_Ncols_and_Nvariants(aux_in)

    # get list of species in the input data
    species_list_of_data=[str(spc).split('-')[1] for spc in data_in[0].grids[0] if 'abs_species' in str(spc)]


    # create FSM-object
    Flxsim = fsm.FluxSimulator(setup_name+'_SW')

    # add absorption species
    abs_species=define_abs_species(Flxsim, species_list_of_data)
    Flxsim.add_species( abs_species, verbose=True)

    Flxsim.set_frequency_grid(f_grid)
    Flxsim.gas_scattering=True
    Flxsim.emission=False

    #Set sun to get sun parameter
    Flxsim.set_sun()
    sun_dist=Flxsim.get_sun()[0].distance*1.


    #calculate LUT
    Flxsim.get_lookuptableBatch(data_in)

    # =============================================================================
    # calculate fluxes
    index=0

    #len of atmospheres, levels and frequencies
    n_atms=len(data_in)
    n_levels=len(data_in[index].grids[1].value)
    n_freqs=len(f_grid)

    #Allocate result arrays
    Result={}
    Result['altitude']=np.zeros((N_variants, N_cols,n_levels))
    Result['pressure']=np.zeros((N_variants, N_cols,n_levels))
    Result['flux_clearsky_up']=np.zeros((N_variants, N_cols,n_levels))
    Result['flux_clearsky_down']=np.zeros((N_variants, N_cols,n_levels))
    Result['spectral_flux_up_TOA']=np.zeros((N_variants, N_cols,n_freqs))    
    Result['spectral_flux_down_TOA']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_up_SFC']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_down_SFC']=np.zeros((N_variants, N_cols,n_freqs))    
    Result['index']=np.arange(0,n_atms)


    #index solar_zenith_angle
    idx_sza=[idx for idx, gr in enumerate(aux_in[0].grids[0]) if 'solar_zenith_angle' == str(gr)][0]

    #index tota_solar_ittadiance
    idx_tsi=[idx for idx, gr in enumerate(aux_in[0].grids[0]) if 'total_solar_irradiance' == str(gr)][0]

    #index surface_emissivity
    idx_surf_emiss=[idx for idx, gr in enumerate(aux_in[0].grids[0]) if 'surface_emissivity' == str(gr)][0]

    #index surface_temperature
    idx_surf_temp=[idx for idx, gr in enumerate(aux_in[0].grids[0]) if 'surface_temperature' == str(gr)][0]


    for index in range(len(data_in)):

        print(f'index: {index}')

        #atmosphere
        atm=data_in[index]
        aux=aux_in[index]

        #surface vavariables
        surface_temperature=aux.data[idx_surf_temp]
        surface_altitude=atm.data[1,0,0,0]
        surface_reflectivity_sw=1-aux.data[idx_surf_emiss]


        #Solar zenith angle and total solar irradiance of scenario
        sza=aux.data[idx_sza]
        if sza>90:
            print(f'index: {index} - szs:{sza}')


        tsi=aux.data[idx_tsi]

        #set sun position according to Sect 4.1 of star arts paper
        Toa_altitude = Flxsim.ws.refellipsoid.value[0]+atm[1,-1,0,0]
        phi=sza-np.rad2deg(np.arcsin(Toa_altitude/sun_dist*np.sin(np.pi-np.deg2rad(sza))))

        Flxsim.ws.sunsChangeGeometry(distance=sun_dist, latitude=0, longitude=phi, index=0)
        Flxsim.scale_sun_to_specific_TSI_at_TOA(tsi, 0, 0, atm[1,-1,0,0])


        results = Flxsim.flux_simulator_single_profile(
            atm,
            surface_temperature,
            surface_altitude,
            surface_reflectivity_sw,
            geographical_position=[0,0],
        )
        #we set the geographical_position to 0,0 because we want to mimic the RFMIP sza

        column_index=int(aux.data[idx_col])
        variant_index=int(aux.data[idx_var])

        if reverse_vertical_order:
            Result['altitude'][variant_index,column_index,:]=results['altitude'][::-1]
            Result['pressure'][variant_index,column_index,:]=results['pressure'][::-1]
            Result['flux_clearsky_up'][variant_index,column_index,:]=results['flux_clearsky_up'][::-1]
            Result['flux_clearsky_down'][variant_index,column_index,:]=results['flux_clearsky_down'][::-1]
        else:
            Result['altitude'][variant_index,column_index,:]=results['altitude']
            Result['pressure'][variant_index,column_index,:]=results['pressure']
            Result['flux_clearsky_up'][variant_index,column_index,:]=results['flux_clearsky_up']
            Result['flux_clearsky_down'][variant_index,column_index,:]=results['flux_clearsky_down']
        Result['spectral_flux_up_TOA'][variant_index,column_index,:]=results['spectral_flux_clearsky_up'][:,-1]
        Result['spectral_flux_down_TOA'][variant_index,column_index,:]=results['spectral_flux_clearsky_down'][:,-1]
        Result['spectral_flux_up_SFC'][variant_index,column_index,:]=results['spectral_flux_clearsky_up'][:,0]
        Result['spectral_flux_down_SFC'][variant_index,column_index,:]=results['spectral_flux_clearsky_down'][:,0]

    #Export results to NetCDF
    results_folder_SW=results_folder / 'SW'
    ds = export_to_xarray(Result, N_variants, N_cols, n_levels, n_freqs, f_grid, results_folder_SW, export_results)


    return ds, Flxsim

def rte_benchmark_lw(data_in, aux_in, f_grid, results_folder, setup_name, export_results=True, reverse_vertical_order=True):
    """Compute longwave radiative transfer using FluxSimulator.

    Parameters
    ----------
    data_in : list
        List of atmospheric data objects.
    aux_in : list
        List of auxiliary data objects (surface properties, etc.).
    f_grid : array
        Frequency grid in Hz.
    results_folder : str
        Path to folder for saving results.
    setup_name : str
        Name of the setup/scenario.
    export_results : bool, optional
        Whether to export results to NetCDF file (default: True).
    reverse_vertical_order : bool, optional
        Whether to reverse vertical order of output arrays (default: True).
        This is the order of the RFMIP data, which is from top to bottom,
        while the output of the FluxSimulator is from bottom to top.

    Returns
    -------
    tuple
        (ds, LW_flxsim) where:
        - ds : xr.Dataset - xarray dataset with computed fluxes
        - LW_flxsim : FluxSimulator - Longwave FluxSimulator object
    """
    # get number of columns and variants in the input data
    N_cols, N_variants, idx_col, idx_var = get_Ncols_and_Nvariants(aux_in)

    # get list of species in the input data
    species_list_of_data=[str(spc).split('-')[1] for spc in data_in[0].grids[0] if 'abs_species' in str(spc)]


    # create FSM-object
    Flxsim = fsm.FluxSimulator(setup_name+'_LW')

    # add absorption species
    abs_species=define_abs_species(Flxsim, species_list_of_data)
    Flxsim.add_species( abs_species, verbose=True)

    Flxsim.set_frequency_grid(f_grid)
    Flxsim.gas_scattering=False
    Flxsim.emission=True

    #calculate LUT
    Flxsim.get_lookuptableBatch(data_in)

    # =============================================================================
    # calculate fluxes
    index=0

    #len of atmospheres, levels and frequencies
    n_atms=len(data_in)
    n_levels=len(data_in[index].grids[1].value)
    n_freqs=len(f_grid)

    #Allocate result arrays
    Result={}
    Result['altitude']=np.zeros((N_variants, N_cols,n_levels))
    Result['pressure']=np.zeros((N_variants, N_cols,n_levels))
    Result['flux_clearsky_up']=np.zeros((N_variants, N_cols,n_levels))
    Result['flux_clearsky_down']=np.zeros((N_variants, N_cols,n_levels))
    Result['spectral_flux_up_TOA']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_down_TOA']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_up_SFC']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_down_SFC']=np.zeros((N_variants, N_cols,n_freqs))
    Result['index']=np.arange(0,n_atms)

    #index surface_emissivity
    idx_surf_emiss=[idx for idx, gr in enumerate(aux_in[0].grids[0]) if 'surface_emissivity' == str(gr)][0]

    #index surface_temperature
    idx_surf_temp=[idx for idx, gr in enumerate(aux_in[0].grids[0]) if 'surface_temperature' == str(gr)][0]


    for index in range(len(data_in)):

        print(f'index: {index}')

        #atmosphere
        atm=data_in[index]
        aux=aux_in[index]

        #surface vavariables
        surface_temperature=aux.data[idx_surf_temp]
        surface_altitude=atm.data[1,0,0,0]
        surface_reflectivity=1-aux.data[idx_surf_emiss]


        results = Flxsim.flux_simulator_single_profile(
            atm,
            surface_temperature,
            surface_altitude,
            surface_reflectivity,
            geographical_position=[0,0],
        )
        #we set the geographical_position to 0,0 because we want to mimic the RFMIP sza

        column_index=int(aux.data[idx_col])
        variant_index=int(aux.data[idx_var])


        if reverse_vertical_order:
            Result['altitude'][variant_index,column_index,:]=results['altitude'][::-1]
            Result['pressure'][variant_index,column_index,:]=results['pressure'][::-1]
            Result['flux_clearsky_up'][variant_index,column_index,:]=results['flux_clearsky_up'][::-1]
            Result['flux_clearsky_down'][variant_index,column_index,:]=results['flux_clearsky_down'][::-1]

        else:
            Result['altitude'][variant_index,column_index,:]=results['altitude']
            Result['pressure'][variant_index,column_index,:]=results['pressure']
            Result['flux_clearsky_up'][variant_index,column_index,:]=results['flux_clearsky_up']
            Result['flux_clearsky_down'][variant_index,column_index,:]=results['flux_clearsky_down']
        
        Result['spectral_flux_down_TOA'][variant_index,column_index,:]=results['spectral_flux_clearsky_down'][:,-1]
        Result['spectral_flux_up_TOA'][variant_index,column_index,:]=results['spectral_flux_clearsky_up'][:,-1]
        Result['spectral_flux_down_SFC'][variant_index,column_index,:]=results['spectral_flux_clearsky_down'][:,0]        
        Result['spectral_flux_up_SFC'][variant_index,column_index,:]=results['spectral_flux_clearsky_up'][:,0]


    #Export results to xarray
    ressults_folder_LW=results_folder / 'LW'
    ds = export_to_xarray(Result, N_variants, N_cols, n_levels, n_freqs, f_grid, ressults_folder_LW, export_results)
    

    return ds, Flxsim


def calc_distance2matchTOATSI(SW_flxsim, atm, latitude, longitude, sza, tsi):
    """Calculate the distance to match the top-of-atmosphere total solar irradiance.

    Parameters
    ----------
    SW_flxsim : FluxSimulator
        FluxSimulator object with sun parameters.
    atm : array-like
        Atmospheric data for the current column.
    latitude : float
        Latitude of the location.
    longitude : float
        Longitude of the location.
    sza : float
        Solar zenith angle in degrees.
    tsi : float
        Total solar irradiance value to match.

    Returns
    -------
    float
        Calculated distance to match the TOA total solar irradiance.
    """
    
    try:
        len(SW_flxsim.ws.suns.value)
    except:
        print("No sun source defined!")
        print("Please define a sun source first!")
        return 

    #TOA altitude above the reference ellipsoid
    Toa_altitude = atm[1,-1,0,0]

    distance_new=SW_flxsim.get_sun_distance_to_match_specific_TSI(tsi, latitude, longitude, Toa_altitude)

    phi=sza-np.rad2deg(np.arcsin(Toa_altitude/distance_new*np.sin(np.pi-np.deg2rad(sza))))

    return distance_new, phi


def rte_benchmark_batch_sw(atms, auxes, f_grid, results_folder, setup_name, export_results=True, reverse_vertical_order=True):

    #index solar_zenith_angle
    idx_sza=[idx for idx, gr in enumerate(auxes[0].grids[0]) if 'solar_zenith_angle' == str(gr)][0]

    #index tota_solar_ittadiance
    idx_tsi=[idx for idx, gr in enumerate(auxes[0].grids[0]) if 'total_solar_irradiance' == str(gr)][0]

    #index surface_emissivity
    idx_surf_emiss=[idx for idx, gr in enumerate(auxes[0].grids[0]) if 'surface_emissivity' == str(gr)][0]

    #index surface_temperature
    idx_surf_temp=[idx for idx, gr in enumerate(auxes[0].grids[0]) if 'surface_temperature' == str(gr)][0]


    # get list of species in the input data
    species_list_of_data=[str(spc).split('-')[1] for spc in atms[0].grids[0] if 'abs_species' in str(spc)]

    # =============================================================================
    # the simulation
    # =============================================================================

    # setup ARTS
    FlxsimBatch = fsm.FluxSimulator(setup_name+'_Batch_SW')
    FlxsimBatch.set_frequency_grid(f_grid)

    # some data preparations
    surface_altitudes = [atm[1,0,0,0] for atm in atms]
    surface_tempratures = [aux[idx_surf_temp] for aux in auxes]
    geographical_positions = [[0, 0] for aux_i in auxes]
    surface_reflectivities = [[1-aux[idx_surf_emiss]] for aux in auxes]

    #calc solar distance and longitudes to match TSI
    FlxsimBatch.set_sun()
    sun_positions = []

    for aux, atm in zip(auxes, atms):
        sza=aux[idx_sza]
        tsi=aux[idx_tsi]
        distance, phi=calc_distance2matchTOATSI(FlxsimBatch, atm, 0, 0, sza, tsi)        
        sun_positions.append([distance, 0, phi])

    # add absorption species
    abs_species=define_abs_species(FlxsimBatch, species_list_of_data)
    FlxsimBatch.add_species( abs_species, verbose=True)
    
    FlxsimBatch.emission = 0
    FlxsimBatch.gas_scattering = True

    results = FlxsimBatch.flux_simulator_batch(
        atms,
        surface_tempratures,
        surface_altitudes,
        surface_reflectivities,
        geographical_positions,
        sun_positions,
        end_index=-1,
        spectral_output=True
    )

    #len of atmospheres, levels and frequencies
    n_levels=len(atms[0].grids[1].value)
    n_freqs=len(f_grid)

    # get number of columns and variants in the input data
    N_cols, N_variants, idx_col, idx_var = get_Ncols_and_Nvariants(auxes)

    #Allocate result arrays
    Result={}
    Result['altitude']=np.zeros((N_variants, N_cols,n_levels))
    Result['pressure']=np.zeros((N_variants, N_cols,n_levels))
    Result['flux_clearsky_up']=np.zeros((N_variants, N_cols,n_levels))
    Result['flux_clearsky_down']=np.zeros((N_variants, N_cols,n_levels))
    Result['spectral_flux_up_TOA']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_down_TOA']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_up_SFC']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_down_SFC']=np.zeros((N_variants, N_cols,n_freqs))
    Result['index']=np.zeros((N_variants, N_cols), dtype=int)

    # Fill the result arrays with the simulation results
    for i, (atm, aux) in enumerate(zip(atms, auxes)):
        col_index=int(aux[idx_col])
        var_index=int(aux[idx_var])

        if reverse_vertical_order:
            Result['altitude'][var_index, col_index,:]=results['array_of_altitude'][i][::-1]
            Result['pressure'][var_index, col_index,:]=results['array_of_pressure'][i][::-1]
            Result['flux_clearsky_up'][var_index, col_index,:]=results["array_of_flux_clearsky_up"][i][::-1]
            Result['flux_clearsky_down'][var_index, col_index,:]=results["array_of_flux_clearsky_down"][i][::-1]
        else:
            Result['altitude'][var_index, col_index,:]=results['array_of_altitude'][i]
            Result['pressure'][var_index, col_index,:]=results['array_of_pressure'][i]
            Result['flux_clearsky_up'][var_index, col_index,:]=results["array_of_flux_clearsky_up"][i]
            Result['flux_clearsky_down'][var_index, col_index,:]=results["array_of_flux_clearsky_down"][i]
        Result['spectral_flux_up_TOA'][var_index, col_index,:]=results["array_of_spectral_flux_clearsky_up"][i][:,-1]
        Result['spectral_flux_down_TOA'][var_index, col_index,:]=results["array_of_spectral_flux_clearsky_down"][i][:,-1]
        Result['spectral_flux_up_SFC'][var_index, col_index,:]=results["array_of_spectral_flux_clearsky_up"][i][:,0]
        Result['spectral_flux_down_SFC'][var_index, col_index,:]=results["array_of_spectral_flux_clearsky_down"][i][:,0]
        Result['index'][var_index, col_index]=results["array_of_index"][i]

    results_folder_SW=results_folder / 'SW'
    ds=export_to_xarray(Result, N_variants, N_cols, n_levels, n_freqs, f_grid, results_folder_SW, export_results)    

    return ds, FlxsimBatch

def rte_benchmark_batch_lw(atms, auxes, f_grid, results_folder, setup_name, export_results=True, reverse_vertical_order=True):

    #index surface_emissivity
    idx_surf_emiss=[idx for idx, gr in enumerate(auxes[0].grids[0]) if 'surface_emissivity' == str(gr)][0]

    #index surface_temperature
    idx_surf_temp=[idx for idx, gr in enumerate(auxes[0].grids[0]) if 'surface_temperature' == str(gr)][0]


    # get list of species in the input data
    species_list_of_data=[str(spc).split('-')[1] for spc in atms[0].grids[0] if 'abs_species' in str(spc)]

    # =============================================================================
    # the simulation
    # =============================================================================

    # setup ARTS
    FlxsimBatch = fsm.FluxSimulator(setup_name+'_Batch_LW')
    FlxsimBatch.set_frequency_grid(f_grid)

    # some data preparations
    surface_altitudes = [atm[1,0,0,0] for atm in atms]
    surface_tempratures = [aux[idx_surf_temp] for aux in auxes]
    geographical_positions = [[0, 0] for aux_i in auxes]
    surface_reflectivities = [[1-aux[idx_surf_emiss]] for aux in auxes]
    sun_positions = [[] for aux_i in auxes]  

    # add absorption species
    abs_species=define_abs_species(FlxsimBatch, species_list_of_data)
    FlxsimBatch.add_species( abs_species, verbose=True)
    
    FlxsimBatch.emission = 1
    FlxsimBatch.gas_scattering = False

    results = FlxsimBatch.flux_simulator_batch(
        atms,
        surface_tempratures,
        surface_altitudes,
        surface_reflectivities,
        geographical_positions,
        sun_positions,
        end_index=-1,
        spectral_output=True
    )

    #len of atmospheres, levels and frequencies
    n_levels=len(atms[0].grids[1].value)
    n_freqs=len(f_grid)

    # get number of columns and variants in the input data
    N_cols, N_variants, idx_col, idx_var = get_Ncols_and_Nvariants(auxes)

    #Allocate result arrays
    Result={}
    Result['altitude']=np.zeros((N_variants, N_cols,n_levels))
    Result['pressure']=np.zeros((N_variants, N_cols,n_levels))
    Result['flux_clearsky_up']=np.zeros((N_variants, N_cols,n_levels))
    Result['flux_clearsky_down']=np.zeros((N_variants, N_cols,n_levels))
    Result['spectral_flux_up_TOA']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_down_TOA']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_up_SFC']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_down_SFC']=np.zeros((N_variants, N_cols,n_freqs))
    Result['index']=np.zeros((N_variants, N_cols), dtype=int)

    # Fill the result arrays with the simulation results
    for i, (atm, aux) in enumerate(zip(atms, auxes)):
        col_index=int(aux[idx_col])
        var_index=int(aux[idx_var])

        if reverse_vertical_order:
            Result['altitude'][var_index, col_index,:]=results['array_of_altitude'][i][::-1]
            Result['pressure'][var_index, col_index,:]=results['array_of_pressure'][i][::-1]
            Result['flux_clearsky_up'][var_index, col_index,:]=results["array_of_flux_clearsky_up"][i][::-1]
            Result['flux_clearsky_down'][var_index, col_index,:]=results["array_of_flux_clearsky_down"][i][::-1]
        else:
            Result['altitude'][var_index, col_index,:]=results['array_of_altitude'][i]
            Result['pressure'][var_index, col_index,:]=results['array_of_pressure'][i]
            Result['flux_clearsky_up'][var_index, col_index,:]=results["array_of_flux_clearsky_up"][i]
            Result['flux_clearsky_down'][var_index, col_index,:]=results["array_of_flux_clearsky_down"][i]
        Result['spectral_flux_up_TOA'][var_index, col_index,:]=results["array_of_spectral_flux_clearsky_up"][i][:,-1]
        Result['spectral_flux_down_TOA'][var_index, col_index,:]=results["array_of_spectral_flux_clearsky_down"][i][:,-1]
        Result['spectral_flux_up_SFC'][var_index, col_index,:]=results["array_of_spectral_flux_clearsky_up"][i][:,0]
        Result['spectral_flux_down_SFC'][var_index, col_index,:]=results["array_of_spectral_flux_clearsky_down"][i][:,0]
        Result['index'][var_index, col_index]=results["array_of_index"][i]

    results_folder_LW=results_folder / 'LW' 
    ds=export_to_xarray(Result, N_variants, N_cols, n_levels, n_freqs, f_grid, results_folder_LW, export_results)    

    return ds, FlxsimBatch