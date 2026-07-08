#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Manfred Brath
"""

import os
import pathlib
import numpy as np
import pyarts as pa
import FluxSimulator as fsm
import xarray as xr
import rte_benchmarks as rtb



#=============================================================================
# %% define functions
#=============================================================================


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
    abs_species=rtb.define_abs_species(FlxsimBatch, species_list_of_data)
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
    N_cols, N_variants, idx_col, idx_var = rtb.get_Ncols_and_Nvariants(auxes)

    #Allocate result arrays
    Result={}
    Result['altitude']=np.zeros((N_variants, N_cols,n_levels))
    Result['pressure']=np.zeros((N_variants, N_cols,n_levels))
    Result['flux_clearsky_up']=np.zeros((N_variants, N_cols,n_levels))
    Result['flux_clearsky_down']=np.zeros((N_variants, N_cols,n_levels))
    Result['spectral_flux_up_TOA']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_down_SFC']=np.zeros((N_variants, N_cols,n_freqs))
    Result['index']=np.zeros((N_variants, N_cols), dtype=int)

    # Fill the result arrays with the simulation results
    for i, (atm, aux) in enumerate(zip(atms, auxes)):
        col_index=int(aux[idx_col])
        var_index=int(aux[idx_var])

        if reverse_vertical_order:
            Result['altitude'][var_index, col_index,:]=atm[1,:,0,0][::-1]
            Result['pressure'][var_index, col_index,:]=atm[2,:,0,0][::-1]
            Result['flux_clearsky_up'][var_index, col_index,:]=results["array_of_flux_clearsky_up"][i][::-1]
            Result['flux_clearsky_down'][var_index, col_index,:]=results["array_of_flux_clearsky_down"][i][::-1]
        else:
            Result['altitude'][var_index, col_index,:]=atm[1,:,0,0]
            Result['pressure'][var_index, col_index,:]=atm[2,:,0,0]
            Result['flux_clearsky_up'][var_index, col_index,:]=results["array_of_flux_clearsky_up"][i]
            Result['flux_clearsky_down'][var_index, col_index,:]=results["array_of_flux_clearsky_down"][i]
        Result['spectral_flux_up_TOA'][var_index, col_index,:]=results["array_of_spectral_flux_clearsky_up"][i][:,-1]
        Result['spectral_flux_down_SFC'][var_index, col_index,:]=results["array_of_spectral_flux_clearsky_down"][i][:,0]
        Result['index'][var_index, col_index]=results["array_of_index"][i]


    ds=rtb.export_to_xarray(Result, N_variants, N_cols, n_levels, n_freqs, f_grid, results_folder_setup, export_results)    

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
    sun_positions = [[1,0,0] for aux_i in auxes]  # Placeholder for sun positions in longwave

    # add absorption species
    abs_species=rtb.define_abs_species(FlxsimBatch, species_list_of_data)
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
    N_cols, N_variants, idx_col, idx_var = rtb.get_Ncols_and_Nvariants(auxes)

    #Allocate result arrays
    Result={}
    Result['altitude']=np.zeros((N_variants, N_cols,n_levels))
    Result['pressure']=np.zeros((N_variants, N_cols,n_levels))
    Result['flux_clearsky_up']=np.zeros((N_variants, N_cols,n_levels))
    Result['flux_clearsky_down']=np.zeros((N_variants, N_cols,n_levels))
    Result['spectral_flux_up_TOA']=np.zeros((N_variants, N_cols,n_freqs))
    Result['spectral_flux_down_SFC']=np.zeros((N_variants, N_cols,n_freqs))
    Result['index']=np.zeros((N_variants, N_cols), dtype=int)

    # Fill the result arrays with the simulation results
    for i, (atm, aux) in enumerate(zip(atms, auxes)):
        col_index=int(aux[idx_col])
        var_index=int(aux[idx_var])

        if reverse_vertical_order:
            Result['altitude'][var_index, col_index,:]=atm[1,:,0,0][::-1]
            Result['pressure'][var_index, col_index,:]=atm[2,:,0,0][::-1]
            Result['flux_clearsky_up'][var_index, col_index,:]=results["array_of_flux_clearsky_up"][i][::-1]
            Result['flux_clearsky_down'][var_index, col_index,:]=results["array_of_flux_clearsky_down"][i][::-1]
        else:
            Result['altitude'][var_index, col_index,:]=atm[1,:,0,0]
            Result['pressure'][var_index, col_index,:]=atm[2,:,0,0]
            Result['flux_clearsky_up'][var_index, col_index,:]=results["array_of_flux_clearsky_up"][i]
            Result['flux_clearsky_down'][var_index, col_index,:]=results["array_of_flux_clearsky_down"][i]
        Result['spectral_flux_up_TOA'][var_index, col_index,:]=results["array_of_spectral_flux_clearsky_up"][i][:,-1]
        Result['spectral_flux_down_SFC'][var_index, col_index,:]=results["array_of_spectral_flux_clearsky_down"][i][:,0]
        Result['index'][var_index, col_index]=results["array_of_index"][i]


    ds=rtb.export_to_xarray(Result, N_variants, N_cols, n_levels, n_freqs, f_grid, results_folder_setup, export_results)    

    return ds, FlxsimBatch


# =============================================================================
# %% Main
# =============================================================================

if __name__ == "__main__":

    # =============================================================================
    # % paths/constants
    # =============================================================================

    # % Get the directory where this script is located
    script_dir = pathlib.Path(__file__).parent.resolve()

    data_folder=script_dir.parent / "data/"


    # level_input_name='rte-examples-arts_rfmip-states.xml'
    # aux_input_name='rte-examples-arts_aux_rfmip-states.xml'

    results_folder=script_dir.parent / "results_test/"

    #Define frequencies (wavenumbers) longwave
    #wavenumber range taken from DDQ paper
    wvn_min_lw=10. # cm^-1
    wvn_max_lw=1/2e-6/100  #cm^-1
    N_wvn_lw=100
    wvn_lw=np.linspace(wvn_min_lw,wvn_max_lw, N_wvn_lw)
    f_grid_lw=pa.arts.convert.kaycm2freq(wvn_lw)

    #Define frequencies (wavenumbers) shortwave
    #wavenumber range taken from DDQ paper
    wvn_min_sw=1/1e-5/100
    wvn_max_sw=1e5
    N_wvn_sw=101
    wvn_sw=np.linspace(wvn_min_sw,wvn_max_sw, N_wvn_sw)
    f_grid_sw=pa.arts.convert.kaycm2freq(wvn_sw)

    # =============================================================================
    # % load data
    # =============================================================================

    #get the list with the input data in the data folder
    data_files = list(data_folder.glob("*.xml"))

    #get the setup name from the first file
    setups=[str(df.stem).split('-')[3] for df in data_files]
    setups=list(set(setups))
    setups.sort()

    setup=setups[1]

    #get the atm file and aux file for the setup
    atm_data_name=f'rte-examples-arts_atm-{setup}.xml'
    aux_data_name=f'rte-examples-arts_aux-{setup}.xml'

    atms=pa.xml.load(str(data_folder/atm_data_name))
    auxes=pa.xml.load(str(data_folder/aux_data_name))

    results_folder_setup=results_folder / f'{setup}/'


    ds_test_sw, fsm_test_sw=rte_benchmark_batch_sw(atms, auxes, f_grid_sw, results_folder, setup)
    ds_test_lw, fsm_test_lw=rte_benchmark_batch_lw(atms, auxes, f_grid_lw, results_folder, setup)