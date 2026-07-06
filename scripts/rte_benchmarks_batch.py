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


#=============================================================================
# %% define functions
#=============================================================================


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
    N_wvn_lw=5000
    wvn_lw=np.linspace(wvn_min_lw,wvn_max_lw, N_wvn_lw)
    f_grid_lw=pa.arts.convert.kaycm2freq(wvn_lw)

    #Define frequencies (wavenumbers) shortwave
    #wavenumber range taken from DDQ paper
    wvn_min_sw=1/1e-5/100
    wvn_max_sw=1e5
    N_wvn_sw=5001
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

    data_in=pa.xml.load(str(data_folder/atm_data_name))
    aux_in=pa.xml.load(str(data_folder/aux_data_name))

    results_folder_setup=results_folder / f'{setup}/'
    os.makedirs(results_folder_setup, exist_ok=True)



    #index solar_zenith_angle
    idx_sza=[idx for idx, gr in enumerate(aux_in[0].grids[0]) if 'solar_zenith_angle' == str(gr)][0]

    #index tota_solar_ittadiance
    idx_tsi=[idx for idx, gr in enumerate(aux_in[0].grids[0]) if 'total_solar_irradiance' == str(gr)][0]

    #index surface_emissivity
    idx_surf_emiss=[idx for idx, gr in enumerate(aux_in[0].grids[0]) if 'surface_emissivity' == str(gr)][0]

    #index surface_temperature
    idx_surf_temp=[idx for idx, gr in enumerate(aux_in[0].grids[0]) if 'surface_temperature' == str(gr)][0]





    # some data preparations
    surface_altitudes = [aux_i[1] for aux_i in aux_in]
    surface_tempratures = [aux_i[0] for aux_i in aux_in]
    geographical_positions = [[aux_i[4], aux_i[5]] for aux_i in aux_in]
    sun_positions = [[1.495978707e11, 0.0, -120.0] for aux_i in aux_in]
    refls = [[0.3] for i in range(len(aux_in))]


    # =============================================================================
    # the simulation
    # =============================================================================

    # setup ARTS
    FluxSimulator_batch = fsm.FluxSimulator("BATCH_Test")
    FluxSimulator_batch.set_frequency_grid(f_grid_sw)
    FluxSimulator_batch.emission = 0
    FluxSimulator_batch.gas_scattering = True
    FluxSimulator_batch.set_species(
        [
            "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
            "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
            "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
            "CO2, CO2-CKDMT252",
            "O3",
            "O3-XFIT",
        ]
    )


    results = FluxSimulator_batch.flux_simulator_batch(
        atms,
        surface_tempratures,
        surface_altitudes,
        refls,
        geographical_positions,
        sun_positions,
        end_index=5,
    )

