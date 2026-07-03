#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Manfred Brath
"""

import os
import pathlib
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


import pyarts as pa

import FluxSimulator as fsm
import xarray as xr

# =============================================================================
# %% paths/constants
# =============================================================================

# %% Get the directory where this script is located
script_dir = pathlib.Path(__file__).parent.resolve()

data_folder=script_dir.parent / "data/"


level_input_name='rte-examples-arts_rfmip-states.xml'
aux_input_name='rte-examples-arts_aux_rfmip-states.xml'

results_folder=script_dir.parent / "../results/rfmip_rerun/"

# =============================================================================
# %% load data
# =============================================================================


data_in=pa.xml.load(str(data_folder/level_input_name))
aux_in=pa.xml.load(str(data_folder/aux_input_name))


def get_Ncols_and_Nvariants(aux_in):
        
    #find index of columns and variants
    grids=aux_in[0].grids[0]
    idx_col=[i for i, gr in enumerate(grids) if 'column_index' in str(gr)][0]
    idx_var=[i for i, gr in enumerate(grids) if 'variant_index' in str(gr)][0]

    cols=[aux[idx_col] for aux in aux_in]
    variants=[aux[idx_var] for aux in aux_in]

    N_cols=int(np.max(cols)+1)
    N_variants=int(np.max(variants)+1)

    return N_cols, N_variants, idx_col, idx_var


N_cols, N_variants, idx_col, idx_var = get_Ncols_and_Nvariants(aux_in)


min_wavelength_sw = 115.5e-9 # [m]
max_wavelength_sw = 9999.5e-9  # [m]
# n_freq_sw=32768*2
n_freq_sw=100

lamb=np.linspace(min_wavelength_sw, max_wavelength_sw,n_freq_sw)
freq_temp=pa.arts.convert.wavelen2freq(lamb)
f_grid_sw=freq_temp[::-1]

species_list_of_data=[str(spc).split('-')[1] for spc in data_in[0].grids[0] if 'abs_species' in str(spc)]

# =============================================================================
# %% create FSM-object
# =============================================================================

SW_flxsim = fsm.FluxSimulator('RFMIP_SW')
SW_flxsim.set_frequency_grid(f_grid_sw)
SW_flxsim.gas_scattering=True
SW_flxsim.emission=False


#set abs_species
abs_species=[]

for i, spc in enumerate(species_list_of_data):
    
    if spc =='O3':
        print('halt')
    
    temp=[spc_arts for spc_arts in SW_flxsim.line_species_available if spc_arts == spc]
    
    if len(temp)==1:
        abs_species.append(spc)
        
    temp=[spc_arts for spc_arts in SW_flxsim.xsec_species_available if spc_arts == spc]        
    
    if len(temp)==1:
        abs_species.append(spc+'-XFIT')

#Set sun to get sun parameter
SW_flxsim.set_sun()
sun_dist=SW_flxsim.get_sun()[0].distance*1.


# #add species
SW_flxsim.add_species( abs_species, verbose=True)

#calculate LUT
SW_flxsim.get_lookuptableBatch(data_in)

# =============================================================================
# %%calculate fluxes
# =============================================================================

index=0

n_atms=len(data_in)
n_levels=len(data_in[index].grids[1].value)
n_freqs=len(f_grid_sw)

Result={}
Result['altitude']=np.zeros((N_variants, N_cols,n_levels))
Result['pressure']=np.zeros((N_variants, N_cols,n_levels))
Result['flux_clearsky_up']=np.zeros((N_variants, N_cols,n_levels))
Result['flux_clearsky_down']=np.zeros((N_variants, N_cols,n_levels))
Result['spectral_flux_up_TOA']=np.zeros((N_variants, N_cols,n_freqs))
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
    sza=aux.data[2]
    if sza>90:
        print(f'index: {index} - szs:{sza}')


    tsi=aux.data[idx_tsi]

    #set sun position according to Sect 4.1 of star arts paper
    Toa_altitude = SW_flxsim.ws.refellipsoid.value[0]+atm[1,-1,0,0]
    phi=sza-np.rad2deg(np.arcsin(Toa_altitude/sun_dist*np.sin(np.pi-np.deg2rad(sza))))
    sun_pos= [sun_dist, 0, phi]

    SW_flxsim.ws.sunsChangeGeometry(distance=sun_dist, latitude=0, longitude=phi, index=0)
    SW_flxsim.scale_sun_to_specific_TSI_at_TOA(tsi, 0, 0, atm[1,-1,0,0])


    results_sw = SW_flxsim.flux_simulator_single_profile(
        atm,
        surface_temperature,
        surface_altitude,
        surface_reflectivity_sw,
        geographical_position=[0,0],
    )
    #we set the geographical_position to 0,0 because we want to mimic the RFMIP sza

    column_index=int(aux.data[idx_col])
    variant_index=int(aux.data[idx_var])

    Result['altitude'][variant_index,column_index,:]=results_sw['altitude']
    Result['pressure'][variant_index,column_index,:]=results_sw['pressure']
    Result['flux_clearsky_up'][variant_index,column_index,:]=results_sw['flux_clearsky_up']
    Result['flux_clearsky_down'][variant_index,column_index,:]=results_sw['flux_clearsky_down']
    Result['spectral_flux_up_TOA'][variant_index,column_index,:]=results_sw['spectral_flux_clearsky_up'][:,-1]
    Result['spectral_flux_down_SFC'][variant_index,column_index,:]=results_sw['spectral_flux_clearsky_down'][:,0]

# Create xarray dataset
ds = xr.Dataset(
    {
        'altitude': (['variant','column','level'], Result['altitude'], {'   units': 'm'}),
        'pressure': (['variant','column','level'], Result['pressure'], {'units': 'Pa'}),
        'flux_clearsky_up': (['variant','column','level'], Result['flux_clearsky_up'], {'units': 'W/m^2'}),
        'flux_clearsky_down': (['variant','column','level'], Result['flux_clearsky_down'], {'units': 'W/m^2'}),
        'spectral_flux_up_TOA': (['variant','column','frequency'], Result['spectral_flux_up_TOA'], {'units': 'W/m^2/Hz'}),
        'spectral_flux_down_SFC': (['variant','column','frequency'], Result['spectral_flux_down_SFC'], {'units': 'W/m^2/Hz'}),
    },
    coords={
        'variant': np.arange(N_variants),
        'column': np.arange(N_cols),
        'level': np.arange(n_levels),
        'frequency': (['frequency'], f_grid_sw, {'units': 'Hz'}),
    }
)

