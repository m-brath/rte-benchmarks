#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:19:27 2026

@author: m-brath

Convert rte-examples netCDF data into ARTS gridded field XML.

This script reads netCDF files from the `rte-examples/` directory and
converts atmospheric profile and species data into ARTS
`ArrayOfGriddedField4` binary XML files. It supports datasets with
column and variant dimensions and interpolates layer quantities to level
pressure coordinates.
"""


import pathlib
import numpy as np
import xarray as xr

import pyarts as pa
import rte_aux_functions as raf




#conversion table 
species_conversion_table = {
    'n2': 'N2',
    'o2': 'O2',
    'co2': 'CO2',
    'h2o': 'H2O',
    'o3': 'O3',
    'co': 'CO',
    'cfc12': 'CFC12',
    'co2': 'CO2',
    'hfc23': 'HFC23',
    'cfc11': 'CFC11',
    'n2o': 'N2O',
    'hfc125': 'HFC125',
    'cf4': 'CF4',
    'hfc143a': 'HFC143a',
    'hfc32': 'HFC32',
    'ch4': 'CH4',
    'hfc134a': 'HFC134a'
}

# =============================================================================
# %% function to convert rte-examples to arts format
# =============================================================================

def convert_rte_to_arts(data, save_path):
    """
    Convert a rte-examples dataset to ARTS format and save it.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing `pres_level`, `temp_level`, `pres_layer`, and
        species concentration variables. Variables are mapped using
        `species_conversion_table`.
    save_path : pathlib.Path or str
        Destination path for the output ARTS XML file.
    """

    # get list of variables in the dataset
    variables = list(data.data_vars.keys())


    columns = data.col.values
    levels = data.level.values

    try:
        variants = data.variant.values
    except AttributeError:
        variants = np.array([0])


    p_lev=data['pres_level'].values.astype('double')
    T_lev=data['temp_level'].values.astype('double')

    p_lay=data['pres_layer'].values.astype('double')


    # check what species are present
    species_in_data = [var for var in variables if var in species_conversion_table]
    species_in_data.sort()

    data_arts = [[]]*len(variants)*len(columns)

    for i in range(len(variants)*len(columns)):
        
        variant_i = i // len(columns)
        column_i = i % len(columns)

        if i % len(columns) == 0:
            print(f"Processing variant {variant_i+1}/{len(variants)} and column {column_i+1}/{len(columns)}")

        data_temp=np.zeros((len(levels), len(species_in_data)+2))
        data_temp[:, 0] = p_lev[column_i,:]
        if len(variants) > 1:
            data_temp[:, 0] = T_lev[variant_i, column_i,:]
        else:
            data_temp[:, 0] = T_lev[column_i,:]

        for j, species in enumerate(species_in_data):

            temp_data = data[species].values.astype('double')
            
            if len(variants) > 1:            
                if temp_data.ndim == 3:
                    temp_data = temp_data[variant_i, column_i, :]
                    data_temp[:, j+2] = raf.lin_interp(np.log(p_lev[column_i,:]), np.log(p_lay[column_i,:]), temp_data,)

                elif temp_data.ndim == 1:
                    temp_data = temp_data[variant_i]
                    data_temp[:, j+2] = np.ones_like(p_lev[column_i,:]) * temp_data


            else:
                if temp_data.ndim == 2:
                    temp_data = temp_data[column_i, :]
                    data_temp[:, j+2] = raf.lin_interp(np.log(p_lev[column_i,:]), np.log(p_lay[column_i,:]), temp_data,)

                elif temp_data.ndim == 1:
                    data_temp[:, j+2] = np.ones_like(p_lev[column_i,:]) * temp_data[0]
            

        temp = raf.make_gridded_field(data_temp, species_in_data, p_lev[column_i,:])
        data_arts[i] = temp

    data_arts=pa.arts.ArrayOfGriddedField4(data_arts)        

    data_arts.savexml(str(save_path),'binary')


# =============================================================================
# %% main
# =============================================================================

if __name__ == "__main__":

    # %% Get the directory where this script is located
    script_dir = pathlib.Path(__file__).parent.resolve()

    # =============================================================================
    # %% constants / paths
    # =============================================================================

    data_dir = script_dir.parent / "../rte-examples/"
    data_files = list(data_dir.glob("*.nc"))

    # =============================================================================
    # %% load data
    # =============================================================================

    for i, df in enumerate(data_files):
        
        print(f"Processing file: {df.name}")
        print(f"File {i+1}/{len(data_files)}")        

        data = xr.open_mfdataset(df, engine="netcdf4")
        name=df.stem

        convert_rte_to_arts(data, save_path=script_dir.parent / "data" / f"rte-examples-arts_{name}.xml")