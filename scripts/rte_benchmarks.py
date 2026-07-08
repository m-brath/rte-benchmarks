#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Manfred Brath
"""

import os
import pathlib
import numpy as np
from pyarts import xml, arts    
import rte_aux_functions as raf



# =============================================================================
# %% paths/constants
# =============================================================================

# % Get the directory where this script is located
script_dir = pathlib.Path(__file__).parent.resolve()

data_folder=script_dir.parent / "data/"


# level_input_name='rte-examples-arts_rfmip-states.xml'
# aux_input_name='rte-examples-arts_aux_rfmip-states.xml'

results_folder=script_dir.parent / "results/"

#Define frequencies (wavenumbers) longwave
#wavenumber range taken from DDQ paper
wvn_min_lw=10. # cm^-1
wvn_max_lw=1/2e-6/100  #cm^-1
N_wvn_lw=100
wvn_lw=np.linspace(wvn_min_lw,wvn_max_lw, N_wvn_lw)
f_grid_lw=arts.convert.kaycm2freq(wvn_lw)

#Define frequencies (wavenumbers) shortwave
#wavenumber range taken from DDQ paper
wvn_min_sw=1/1e-5/100
wvn_max_sw=1e5
N_wvn_sw=101
wvn_sw=np.linspace(wvn_min_sw,wvn_max_sw, N_wvn_sw)
f_grid_sw=arts.convert.kaycm2freq(wvn_sw)


# =============================================================================
# %% load data
# =============================================================================

#get the list with the input data in the data folder
data_files = list(data_folder.glob("*.xml"))

#get the setup name from the first file
setups=[str(df.stem).split('-')[3] for df in data_files]
setups=list(set(setups))
setups.sort()

# =============================================================================
# %% run simulations for each setup
# =============================================================================

for setup in setups:
    print(f'Processing setup: {setup}')

    #get the atm file and aux file for the setup
    atm_data_name=f'rte-examples-arts_atm-{setup}.xml'
    aux_data_name=f'rte-examples-arts_aux-{setup}.xml'

    data_in=xml.load(str(data_folder/atm_data_name))
    aux_in=xml.load(str(data_folder/aux_data_name))

    results_folder_setup=results_folder / f'{setup}/'
    os.makedirs(results_folder_setup, exist_ok=True)

    _, _ = raf.rte_benchmark_sw(data_in, aux_in, f_grid_sw, results_folder_setup, setup, export_results=True)
    _, _ = raf.rte_benchmark_lw(data_in, aux_in, f_grid_lw, results_folder_setup, setup, export_results=True)