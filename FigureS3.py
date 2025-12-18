# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 08:10:35 2025

@author: Harry J Grosvenor

"""

#%% 1. Load Packages

import numpy as np
import xarray as xa
import pandas as pd
import dask.array as da
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cmocean

#%% 2. Find and Load Data

# Global Constants
rho = 1025  # reference density for seawater
cp = 4000   # specific heat of seawater
R_earth = 6.371e6 # Earth's radius in meters

# Define the list of models:
Ensemble = ["CCSM4", "CCSM4-Utrecht", "CESM1.2", "CESM2", "COSMOS", "EC-Earth3-LR",
            "GISS-E2-1-G", "HadCM3", "HadGEM3", "IPSL-CM6A-LR"]

Ensemblef = ["CESM2", "EC-Earth3-LR", "GISS-E2-1-G", "HadGEM3", "IPSL-CM6A-LR"]

# Experiment nomenclature:
# E280 = piControl
# Eoi400 = midPliocene
Experiments = ["E280", "Eoi400"]

Experimentsf = ["control", "historical", "ssp126", "ssp245", "ssp585"]

# Initialize a dictionary to store datasets for each experiment
datasets = {model: {} for model in Ensemble}
datasetsf = {model: {} for model in Ensemble}

# Loop through each experiment
for model in Ensemble:
    # Loop through each model within the experiment
    for experiment in Experiments:
        if experiment == "E280":
            # Construct file path for the current experiment and model
            file_path = f"Data/01.thetaO/01.OneTime/01.Native/{experiment}_{model}_thetao.nc"
            # Read in dataset for the current experiment and model
            datasets[model][experiment] = xa.open_dataset(file_path)
        elif experiment == "Eoi400":
            # Construct file path for the current experiment and model
            file_path = f"Data/01.thetaO/04.Annuals/{experiment}_{model}_thetao_annual.nc"
            # Read in dataset for the current experiment and model
            datasets[model][experiment] = xa.open_dataset(file_path)
    
# Loop through each future experiment
for model in Ensemblef:
    # Loop through each model within the experiment
    for experiment in Experimentsf:
        if experiment == "control":
            # Construct file path for the current experiment and model
            file_path = f"Data/01.thetaO/03.Futures/01.Native/historical_1850_1900_{model}_thetao.nc"
            # Read in dataset for the current experiment and model
            datasetsf[model][experiment] = xa.open_dataset(file_path)
        else:
            # Construct file path for the current experiment and model
            file_path = f"D:/202510.Annual_thetao/ScenarioMIP/{experiment}_{model}_thetao_annual.nc"
            # Read in dataset for the current experiment and model
            datasetsf[model][experiment] = xa.open_dataset(file_path)     
    
#%% 3. Variables

# Initialise empty Thetas dictionary to store potential temperature data
Thetas = {model: {} for model in Ensemble}
lats = {model: {} for model in Ensemble}
lons = {model: {} for model in Ensemble}
depths = {model: {} for model in Ensemble}
times = {model: {} for model in Ensemble}

Thetasf = {model: {} for model in Ensemblef}

# Assign the correct variable name using our pre-defined function
for model in Ensemble:
    for experiment in Experiments:
        # Define thetao variable for model experiments
        Thetas[model][experiment] = datasets[model][experiment].thetao
        # Eliminate unnecessary dimension of size 1 in Theta
        Thetas[model][experiment] = np.squeeze(Thetas[model][experiment])

for model in Ensemblef:
    for experiment in Experimentsf:
        Thetasf[model][experiment] = datasetsf[model][experiment].thetao
        Thetasf[model][experiment] = np.squeeze(Thetasf[model][experiment])

# Lat, Lon and Depth will all have the same grid structure between experiments so only need one array:        
for model in Ensemble:
    lats[model] = datasets[model]['E280'].lat
    lons[model] = datasets[model]['E280'].lon
    depths[model] = datasets[model]['Eoi400'].lev
    times[model] = datasets[model]['Eoi400'].time

latsf = {}
lonsf = {}

for model in Ensemblef:
    latsf[model] = datasetsf[model]['historical'].lat
    lonsf[model] = datasetsf[model]['historical'].lon


# Initialise empty areacello dictionary
areacello = {model: {} for model in Ensemble}

# Loop through each model
for model in Ensemble:
    if lats[model].ndim == 2 and lons[model].ndim == 2:
        # Check if both Lats and Lons have 2D dimensions for the current model
        if model in ["CCSM4", "CESM1.2", "CESM2"]:
            # Use the specific file for these models and experiment
            areacello_file = "Data/10.areacello/areacello_CESM2_midPliocene.nc"
            areacello_ds = xa.open_dataset(areacello_file)
            areacello[model] = areacello_ds.areacello.values
        else:
            areacello_File = f"Data/10.areacello/areacello_{model}_piControl.nc"
            # Open the dataset
            areacello_ds = xa.open_dataset(areacello_File)
            # Add the area cell values to the dictionary for the current model
            areacello[model] = areacello_ds.areacello.values
            
fareacello = {model: {} for model in Ensemblef}

# Loop through each model
for model in Ensemblef:
    if latsf[model].ndim == 2 and lonsf[model].ndim == 2:
            fareacello_File = f"Data/10.areacello/areacello_{model}_piControl.nc"
            # Open the dataset
            fareacello_ds = xa.open_dataset(fareacello_File)
            # Add the area cell values to the dictionary for the current model
            fareacello[model] = fareacello_ds.areacello.values

#%% 4. Theta Anoms

Theta_anoms = {}

for model in Ensemble:
    # Get Eoi400 (4D: time, depth, lat, lon)
    eo = Thetas[model]["Eoi400"]
    # Get E280 (3D: depth, lat, lon)
    pc = Thetas[model]["E280"]
    # Ensure they're using dask (important for memory!)
    eo = eo.chunk({"time": 1})       # process one timestep at a time
    pc = pc.chunk({"depth": -1})       # depth as full slice, lats/lons auto
    # Subtract with manual broadcasting (no alignment on coords)
    Theta_anoms[model] = eo - pc.data  # prevents time expansion of E280
#%%% 4.1 Anomalies for future

Thetaf_anoms = {}
timesf = {model: {} for model in Ensemblef}

for model in Ensemblef:
    ctl = Thetasf[model]['control'].chunk({'depth': -1})
    hist = Thetasf[model]['historical'].chunk({'time': 1})   # one time step per chunk
    Thetaf_anoms[model] = {}
    for experiment in ['ssp126', 'ssp245', 'ssp585']:
        exp = Thetasf[model][experiment].chunk({'time': 1})
        # compute anomalies separately (lazy dask operations)
        hist_anom = hist - ctl.data   # ctl.data will be a dask array because ctl was chunked
        exp_anom  = exp  - ctl.data
        # now concatenate the *anomalies* (still lazy)
        exp_full_anom = xa.concat([hist_anom, exp_anom], dim='time')
        # store the lazy result
        Thetaf_anoms[model][experiment] = exp_full_anom
        # Determine total time length for this model/experiment
        n_time = exp_full_anom.sizes['time']
        # Define time index from 0 to n_time-1
        timesf[model][experiment] = np.arange(n_time)


#%% 5. Calculate grid cell area for each model
# First big processing batch

# Depth is always 1D so can be defined outside the if loop.
Depth_ext = {model: {} for model in Ensemble}
Depth_edges = {model: {} for model in Ensemble}
dDepth = {model: {} for model in Ensemble}

# Define Depth_edges for each experiment
for model in Ensemble:
    if 'z_bounds' in datasets[model]['E280'].variables:
        # Use z_bounds to calculate Depth_ext and Depth_edges
        z_bounds = datasets[model]['E280']['z_bounds'].values  # Shape: (nz, 2)
        upper_bounds = z_bounds[:, 0]  # Extract upper bounds (shallower)
        lower_bounds = z_bounds[:, 1]  # Extract lower bounds (deeper)
        # Depth_ext: Use upper_bounds and extend using the spacing of the last grid cell
        last_spacing = lower_bounds[-1] - upper_bounds[-1]
        Depth_ext[model] = np.append(upper_bounds, upper_bounds[-1] + last_spacing)
        # Depth_edges: Include all upper_bounds and the final lower_bound
        Depth_edges[model] = np.append(upper_bounds, lower_bounds[-1])   
        # dDepth: Differences between consecutive depth edges
        dDepth[model] = Depth_edges[model][1:] - Depth_edges[model][:-1]
    else:
        Depth_ext[model] = np.concatenate((depths[model], [depths[model][-1] + (depths[model][-1] - depths[model][-2])]))
        Depth_edges[model] = np.concatenate(([0], ((Depth_ext[model][:-1] + Depth_ext[model][1:]) / 2)))
        dDepth[model] = Depth_edges[model][1:] - Depth_edges[model][:-1]

grid_cell_area = {model: None for model in Ensemble}

# Grid cell area calculations
for model in Ensemble:
    if lats[model].ndim == 1:
        # Regular grid
        lat_edges = np.radians(np.concatenate((lats[model] - 0.5 * np.diff(lats[model])[0], [lats[model][-1] + 0.5 * np.diff(lats[model])[-1]])))
        lon_edges = np.radians(np.concatenate((lons[model] - 0.5 * np.diff(lons[model])[0], [lons[model][-1] + 0.5 * np.diff(lons[model])[-1]]))) 
        # Vectorized area calculation
        dlat = np.abs(np.diff(lat_edges))[:, None]
        dlon = np.abs(np.diff(lon_edges))
        grid_cell_area[model] = (
            R_earth**2 * np.outer(np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]), dlon))
    elif lats[model].ndim == 2:
        # Irregular grid option
        grid_cell_area[model] = areacello[model] 

#%%% 5.1 Grid Cell area for ssps

# Don't need to change depth

fgrid_cell_area = {model: None for model in Ensemblef}

# Grid cell area calculations
for model in Ensemblef:
    if latsf[model].ndim == 1:
        # Regular grid
        lat_edges = np.radians(np.concatenate((latsf[model] - 0.5 * np.diff(latsf[model])[0], [latsf[model][-1] + 0.5 * np.diff(latsf[model])[-1]])))
        lon_edges = np.radians(np.concatenate((lonsf[model] - 0.5 * np.diff(lonsf[model])[0], [lonsf[model][-1] + 0.5 * np.diff(lonsf[model])[-1]]))) 
        # Vectorized area calculation
        dlat = np.abs(np.diff(lat_edges))[:, None]
        dlon = np.abs(np.diff(lon_edges))
        fgrid_cell_area[model] = (
            R_earth**2 * np.outer(np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]), dlon))
    elif latsf[model].ndim == 2:
        # Irregular grid option
        fgrid_cell_area[model] = fareacello[model] 

#%% 6. Calculate Ocean heat content for every grid cell

#%%% 6.1 Ensuring Valid Cells
# i.e. cells where ocean is present

# Define valid theta, for where there 
Thetas_v = {model: ((~np.isnan(grid_cell_area[model])) & (~np.isnan(Theta_anoms[model]))) for model in Ensemble}

# ---- Cell volume (time-independent) ----
grid_cell_area_ext = {}
dDepth_ext = {}
cell_vol = {}

for model in Ensemble:
    # Expand grid cell area across depth
    grid_cell_area_ext[model] = np.repeat(
        grid_cell_area[model][np.newaxis, :, :], len(depths[model]),axis=0)  # shape: (lev, lat, lon)
    # Expand depth thickness across lat/lon
    # Adjust repeating operations based on the new dimensional order
    if lats[model].ndim == 1:
        # Repeat the depth dimension first, followed by latitude, then longitude
        dDepth_ext[model] = np.repeat(np.repeat(dDepth[model][:, np.newaxis, np.newaxis], len(lats[model]), axis=1), len(lons[model]), axis=2)
    else:
        # Adjust for 2D latitude and longitude cases, repeat accordingly
        dDepth_ext[model] = np.repeat(np.repeat(dDepth[model][:, np.newaxis, np.newaxis], len(lats[model]), axis=1), len(lons[model][0]), axis=2)        

grid_cell_area_v = {model: np.zeros_like(grid_cell_area_ext[model]) for model in Ensemble}
dDepth_v = {model: np.zeros_like(dDepth_ext[model]) for model in Ensemble}

# Populate dictionaries with valid values
for model in Ensemble:
    valid_indices = Thetas_v[model][0, :, :, :] # Assume this is already a boolean array   
    # Start with full grid cell area and depth arrays
    grid_cell_area_v[model] = grid_cell_area_ext[model].copy()  
    dDepth_v[model] = dDepth_ext[model].copy()  
    # Apply the valid_indices mask along the latitude and longitude dimensions
    grid_cell_area_v[model] = np.ma.masked_where(~valid_indices, grid_cell_area_v[model])
    dDepth_v[model] = np.ma.masked_where(~valid_indices, dDepth_v[model])

#%%% 6.2 Running loop

OHC_cells = {}
OHC_m2 = {}
OHC_glob = {}
total_vol = {}
OHC_cells_2D = {}
OHC_m2_2D = {}
OHC_per_m3 = {}
OHC_mn ={}

for model in Ensemble:
    # Final cell volume (no time dimension here)
    cell_vol[model] = grid_cell_area_v[model] * dDepth_v[model]
    # ---- OHC ----
    OHC_cells[model] = rho * cp * Theta_anoms[model] * cell_vol[model]
    OHC_m2[model] = rho * cp * Theta_anoms[model] * dDepth_v[model]
    # ---- Aggregates ----
    OHC_glob[model] = OHC_cells[model].sum(axis=(1,2,3))
    OHC_cells_2D[model] = OHC_cells[model].sum(axis=(1))
    OHC_m2_2D[model] = OHC_m2[model].sum(axis=(1))
    total_vol[model] = cell_vol[model].sum()
    OHC_per_m3[model] = OHC_glob[model] / total_vol[model]

#%%% 6.3 Similar but for SSPs
# i.e. cells where ocean is present

# Define valid theta, for where there 
fThetas_v = {model: ((~np.isnan(fgrid_cell_area[model])) & (~np.isnan(Thetaf_anoms[model]['ssp126']))) for model in Ensemblef}

# ---- Cell volume (time-independent) ----
fgrid_cell_area_ext = {}
fdDepth_ext = {}
fcell_vol = {}

for model in Ensemblef:
    # Expand grid cell area across depth
    fgrid_cell_area_ext[model] = np.repeat(
        fgrid_cell_area[model][np.newaxis, :, :], len(depths[model]),axis=0)  # shape: (lev, lat, lon)
    # Expand depth thickness across lat/lon
    # Adjust repeating operations based on the new dimensional order
    if latsf[model].ndim == 1:
        # Repeat the depth dimension first, followed by latitude, then longitude
        fdDepth_ext[model] = np.repeat(np.repeat(dDepth[model][:, np.newaxis, np.newaxis], len(latsf[model]), axis=1), len(lonsf[model]), axis=2)
    else:
        # Adjust for 2D latitude and longitude cases, repeat accordingly
        fdDepth_ext[model] = np.repeat(np.repeat(dDepth[model][:, np.newaxis, np.newaxis], len(latsf[model]), axis=1), len(lonsf[model][0]), axis=2)        

fgrid_cell_area_v = {model: np.zeros_like(fgrid_cell_area_ext[model]) for model in Ensemblef}
fdDepth_v = {model: np.zeros_like(fdDepth_ext[model]) for model in Ensemblef}

# Populate dictionaries with valid values
for model in Ensemblef:
    valid_indices = fThetas_v[model][0, :, :, :] # Assume this is already a boolean array   
    # Start with full grid cell area and depth arrays
    fgrid_cell_area_v[model] = fgrid_cell_area_ext[model].copy()  
    fdDepth_v[model] = fdDepth_ext[model].copy()  
    # Apply the valid_indices mask along the latitude and longitude dimensions
    fgrid_cell_area_v[model] = np.ma.masked_where(~valid_indices, fgrid_cell_area_v[model])
    fdDepth_v[model] = np.ma.masked_where(~valid_indices, fdDepth_v[model])


fOHC_cells = {model: {} for model in Ensemblef}
fOHC_m2 = {model: {} for model in Ensemblef}
fOHC_glob = {model: {} for model in Ensemblef}
ftotal_vol = {}
fOHC_cells_2D = {model: {} for model in Ensemblef}
fOHC_m2_2D = {model: {} for model in Ensemblef}
fOHC_per_m3 = {model: {} for model in Ensemblef}
fOHC_mn ={model: {} for model in Ensemblef}

for model in Ensemblef:
    for experiment in ['ssp126', 'ssp245', 'ssp585']:
        # Final cell volume (no time dimension here)
        fcell_vol[model] = fgrid_cell_area_v[model] * fdDepth_v[model]
        # ---- OHC ----
        fOHC_cells[model][experiment] = rho * cp * Thetaf_anoms[model][experiment] * fcell_vol[model]
        fOHC_m2[model][experiment] = rho * cp * Thetaf_anoms[model][experiment] * fdDepth_v[model]
        # ---- Aggregates ----
        fOHC_glob[model][experiment] = fOHC_cells[model][experiment].sum(axis=(1,2,3))
        fOHC_cells_2D[model][experiment] = fOHC_cells[model][experiment].sum(axis=(1))
        fOHC_m2_2D[model][experiment] = fOHC_m2[model][experiment].sum(axis=(1))
        ftotal_vol[model] = fcell_vol[model].sum()
        fOHC_per_m3[model][experiment] = fOHC_glob[model][experiment] / ftotal_vol[model]

#%% 7. Fixed Depth Layer

#%%% 7.1 Set chosen boundaries and extents

# Define chosen depth boundaries
# Upper limit
Depth_upp = 0
# Lower limit
Depth_low = 700 # Be careful with maximum depth of file (don't go deeper!)

#%%% 7.2 Calculating Boundaries to be included

# Initialise empty dictionaries for depth calculations
depth_idx_upp_upper = {model: {} for model in Ensemble}
depth_idx_upp_lower = {model: {} for model in Ensemble}
depth_idx_low_upper = {model: {} for model in Ensemble}
depth_idx_low_lower = {model: {} for model in Ensemble}
dDepth_upp_rat = {model: {} for model in Ensemble}
dDepth_low_rat = {model: {} for model in Ensemble}

# Locate the grid cells that the target depths falls within:
for model in Ensemble:
    # Depth
    # upper boundary
    depth_idx_upp_upper[model] = np.where(Depth_edges[model] <= Depth_upp)[0][-1]
    depth_idx_upp_lower[model] = np.where(Depth_edges[model] > Depth_upp)[0][0]
    # lower boundary
    depth_idx_low_upper[model] = np.where(Depth_edges[model] < Depth_low)[0][-1]
    depth_idx_low_lower[model] = np.where(Depth_edges[model] >= Depth_low)[0][0]
    # Calculate the height of these grid cells
    # Depth
    # upper boundary
    dDepth_idx_upp = Depth_edges[model][depth_idx_upp_lower[model]] - Depth_edges[model][depth_idx_upp_upper[model]]
    # lower boundary
    dDepth_idx_low = Depth_edges[model][depth_idx_low_lower[model]] - Depth_edges[model][depth_idx_low_upper[model]]
    # Calculate the height of the new grid level that uses the target depth
    # Depth
    # upper boundary
    dDepth_upp = Depth_edges[model][depth_idx_upp_lower[model]] - Depth_upp
    # lower boundary
    dDepth_low = Depth_low - Depth_edges[model][depth_idx_low_upper[model]]
    # Calculate the ratio between these heights
    # Depth
    # upper boundary
    dDepth_upp_rat[model] = dDepth_upp / dDepth_idx_upp
    # lower boundary
    dDepth_low_rat[model] = dDepth_low / dDepth_idx_low


#%%% 7.3 Calculating OHC between target depths

# Initialise dictionaries
OHC_cells_tar_z = {}
OHC_m2_tar_z    = {}
cell_vol_tar_z  = {}

# Initialise empty dictionaries for storing 2D arrays and summary values
cell_vol_tar_z_glob = {model: {} for model in Ensemble}
cell_vol_tar_z_hov = {model: {} for model in Ensemble}
OHC_m2_tar_z_2D = {model : {} for model in Ensemble}
OHC_tar_z_hov =  {model: {} for model in Ensemble}
OHC_tar_z_glob =  {model: {} for model in Ensemble}
OHC_tar_z_per_m3 =  {model: {} for model in Ensemble}
OHC_tar_z_hov_m3 =  {model: {} for model in Ensemble}

for model in Ensemble:
    # shapes and indices
    nt = len(times[model])
    # number of target depths (including two partials)
    nlev = (depth_idx_low_lower[model] - depth_idx_upp_upper[model])
    # sanity
    if nlev <= 0:
        raise ValueError(f"Bad depth indexing for {model}")
    nlat, nlon = OHC_cells[model].shape[2], OHC_cells[model].shape[3]
    # Ensure OHC source arrays are Dask arrays chunked reasonably
    # tune these chunk sizes to your memory / workers
    target_time_chunk = 10
    target_lev_chunk = 10
    if not isinstance(OHC_cells, da.Array):
        ohc_cells = da.from_array(OHC_cells[model], chunks=(target_time_chunk, target_lev_chunk, nlat, nlon))
    else:
        ohc_cells = OHC_cells[model].rechunk((target_time_chunk, target_lev_chunk, nlat, nlon))
    if not isinstance(OHC_m2[model], da.Array):
        ohc_m2 = da.from_array(OHC_m2[model], chunks=(target_time_chunk, target_lev_chunk, nlat, nlon))
    else:
        ohc_m2 = OHC_m2[model].rechunk((target_time_chunk, target_lev_chunk, nlat, nlon))
    # --- Build 3D cell_vol_tar_z exactly like your logic (3D, no time)
    # create an empty small array (3D) then fill using direct slicing of small in-memory arrays
    cell_vol_tar_z[model] = np.zeros((nlev, nlat, nlon), dtype=cell_vol[model].dtype)
    # indices mapping:
    cell_vol_tar_z[model][1:depth_idx_low_upper[model]-depth_idx_upp_lower[model]+1, :,:] = cell_vol[model][depth_idx_upp_lower[model]:depth_idx_low_upper[model], :, :]
    cell_vol_tar_z[model][0, :, :] = dDepth_upp_rat[model] * cell_vol[model][depth_idx_upp_upper[model], :, :]
    cell_vol_tar_z[model][depth_idx_low_upper[model] - depth_idx_upp_upper[model]-1, :, :] = dDepth_low_rat[model] * cell_vol[model][depth_idx_low_upper[model]-1, :, :]
    # --- Build Dask arrays for OHC components without materialising
    # middle (whole) block: shape (nt, interior_count, nlat, nlon)
    middle_cells = ohc_cells[:, depth_idx_upp_lower[model]:depth_idx_low_upper[model], :, :]  # Dask slice (lazy)
    middle_m2    = ohc_m2[:, depth_idx_upp_lower[model]:depth_idx_low_upper[model], :, :]      # Dask slice
    # top partial: take single depth slice but keep depth axis (1)
    # resulting shape will be (nt, 1, nlat, nlon)
    top_cells = (ohc_cells[:, depth_idx_upp_upper[model], :, :] * dDepth_upp_rat[model])[:, None, :, :]
    top_m2    = (ohc_m2[:, depth_idx_upp_upper[model], :, :] * dDepth_upp_rat[model])[:, None, :, :]
    # bottom partial: similar (nt, 1, nlat, nlon)
    bottom_cells = (ohc_cells[:, depth_idx_low_upper[model]-1, :, :] * dDepth_low_rat[model])[:, None, :, :]
    bottom_m2    = (ohc_m2[:, depth_idx_low_upper[model]-1, :, :] * dDepth_low_rat[model])[:, None, :, :]
    # Concatenate in the depth axis: [top, middle, bottom]
    # The final depth axis length = 1 + interior_count + 1 == nlev
    OHC_cells_tar_z[model] = da.concatenate([top_cells, middle_cells, bottom_cells], axis=1)
    OHC_m2_tar_z[model]    = da.concatenate([top_m2, middle_m2, bottom_m2], axis=1)
    # Rechunk if desired to a stable chunk shape
    OHC_cells_tar_z[model] = OHC_cells_tar_z[model].rechunk((target_time_chunk, target_lev_chunk, nlat, nlon))
    OHC_m2_tar_z[model]    = OHC_m2_tar_z[model].rechunk((target_time_chunk, target_lev_chunk, nlat, nlon))
    # Calculate aggregate values 
    print(f"Fixed layer depth processing for {model} ({nt}, {nlev}, {nlat}, {nlon})")

#%%% 7.4 Calculate summary values

for model in Ensemble:
    # Global volume
    cell_vol_tar_z_glob[model] = np.nansum(cell_vol_tar_z[model], axis=None)
    # Volume per latitude (depth + lon summed)
    cell_vol_tar_z_hov[model] = np.nansum(cell_vol_tar_z[model], axis=(0,2))  # (nlat,)
    # Sum over depth axis (axis=1)
    OHC_m2_tar_z_2D[model] = da.nansum(OHC_m2_tar_z[model], axis=1).compute()
    # Global OHC (time series)
    OHC_tar_z_glob[model] = da.nansum(OHC_cells_tar_z[model], axis=(1,2,3)).compute()  # (nt,)
    # Global mean per m³
    OHC_tar_z_per_m3[model] = OHC_tar_z_glob[model] / cell_vol_tar_z_glob[model]
    print(f"{model}: mean global OHC = {np.nanmean(OHC_tar_z_glob[model])}")


#%%% 7.5 Same for SSPs

# Initialise dictionaries
fOHC_cells_tar_z = {model: {} for model in Ensemblef}
fOHC_m2_tar_z    = {model: {} for model in Ensemblef}
fcell_vol_tar_z  = {model: {} for model in Ensemblef}

# Initialise empty dictionaries for storing 2D arrays and summary values
fcell_vol_tar_z_glob = {model: {} for model in Ensemblef}
fcell_vol_tar_z_hov = {model: {} for model in Ensemblef}
fOHC_m2_tar_z_2D = {model : {} for model in Ensemblef}
fOHC_tar_z_hov =  {model: {} for model in Ensemblef}
fOHC_tar_z_glob =  {model: {} for model in Ensemblef}
fOHC_tar_z_per_m3 =  {model: {} for model in Ensemblef}
fOHC_tar_z_hov_m3 =  {model: {} for model in Ensemblef}

for model in Ensemblef:
    for experiment in ['ssp126', 'ssp245', 'ssp585']:
        # shapes and indices
        nt = len(timesf[model][experiment])
        # number of target depths (including two partials)
        nlev = (depth_idx_low_lower[model] - depth_idx_upp_upper[model])
        # sanity
        if nlev <= 0:
            raise ValueError(f"Bad depth indexing for {model}")
        nlat, nlon = fOHC_cells[model][experiment].shape[2], fOHC_cells[model][experiment].shape[3]
        # Ensure OHC source arrays are Dask arrays chunked reasonably
        # tune these chunk sizes to your memory / workers
        target_time_chunk = 10
        target_lev_chunk = 10
        if not isinstance(fOHC_cells[model], da.Array):
            fohc_cells = da.from_array(fOHC_cells[model][experiment], chunks=(target_time_chunk, target_lev_chunk, nlat, nlon))
        else:
            fohc_cells = fOHC_cells[model][experiment].rechunk((target_time_chunk, target_lev_chunk, nlat, nlon))
        if not isinstance(fOHC_m2[model], da.Array):
            fohc_m2 = da.from_array(fOHC_m2[model][experiment], chunks=(target_time_chunk, target_lev_chunk, nlat, nlon))
        else:
            fohc_m2 = fOHC_m2[model][experiment].rechunk((target_time_chunk, target_lev_chunk, nlat, nlon))
        # --- Build 3D cell_vol_tar_z exactly like your logic (3D, no time)
        # create an empty small array (3D) then fill using direct slicing of small in-memory arrays
        fcell_vol_tar_z[model] = np.zeros((nlev, nlat, nlon), dtype=fcell_vol[model].dtype)
        # indices mapping:
        fcell_vol_tar_z[model][1:depth_idx_low_upper[model]-depth_idx_upp_lower[model]+1, :,:] = fcell_vol[model][depth_idx_upp_lower[model]:depth_idx_low_upper[model], :, :]
        fcell_vol_tar_z[model][0, :, :] = dDepth_upp_rat[model] * fcell_vol[model][depth_idx_upp_upper[model], :, :]
        fcell_vol_tar_z[model][depth_idx_low_upper[model] - depth_idx_upp_upper[model]-1, :, :] = dDepth_low_rat[model] * fcell_vol[model][depth_idx_low_upper[model]-1, :, :]
        # --- Build Dask arrays for OHC components without materialising
        # middle (whole) block: shape (nt, interior_count, nlat, nlon)
        middle_cells = fohc_cells[:, depth_idx_upp_lower[model]:depth_idx_low_upper[model], :, :]  # Dask slice (lazy)
        middle_m2    = fohc_m2[:, depth_idx_upp_lower[model]:depth_idx_low_upper[model], :, :]      # Dask slice
        # top partial: take single depth slice but keep depth axis (1)
        # resulting shape will be (nt, 1, nlat, nlon)
        top_cells = (fohc_cells[:, depth_idx_upp_upper[model], :, :] * dDepth_upp_rat[model])[:, None, :, :]
        top_m2    = (fohc_m2[:, depth_idx_upp_upper[model], :, :] * dDepth_upp_rat[model])[:, None, :, :]
        # bottom partial: similar (nt, 1, nlat, nlon)
        bottom_cells = (fohc_cells[:, depth_idx_low_upper[model]-1, :, :] * dDepth_low_rat[model])[:, None, :, :]
        bottom_m2    = (fohc_m2[:, depth_idx_low_upper[model]-1, :, :] * dDepth_low_rat[model])[:, None, :, :]
        # Concatenate in the depth axis: [top, middle, bottom]
        # The final depth axis length = 1 + interior_count + 1 == nlev
        fOHC_cells_tar_z[model][experiment] = da.concatenate([top_cells, middle_cells, bottom_cells], axis=1)
        fOHC_m2_tar_z[model][experiment]    = da.concatenate([top_m2, middle_m2, bottom_m2], axis=1)
        # Rechunk if desired to a stable chunk shape
        fOHC_cells_tar_z[model][experiment] = fOHC_cells_tar_z[model][experiment].rechunk((target_time_chunk, target_lev_chunk, nlat, nlon))
        fOHC_m2_tar_z[model][experiment]    = fOHC_m2_tar_z[model][experiment].rechunk((target_time_chunk, target_lev_chunk, nlat, nlon))
        # Calculate aggregate values 
        print(f"Fixed layer depth processing for {model}: {experiment} ({nt}, {nlev}, {nlat}, {nlon})")

#%%%% 7.5.1 Calculate summary values

for model in Ensemblef:
    # Global volume
    fcell_vol_tar_z_glob[model] = np.nansum(fcell_vol_tar_z[model], axis=None)
    # Volume per latitude (depth + lon summed)
    fcell_vol_tar_z_hov[model] = np.nansum(fcell_vol_tar_z[model], axis=(0,2))  # (nlat,)
    for experiment in ['ssp126', 'ssp245', 'ssp585']:
        # Sum over depth axis (axis=1)
        fOHC_m2_tar_z_2D[model][experiment] = da.nansum(fOHC_m2_tar_z[model][experiment], axis=1).compute()
        # Global OHC (time series)
        fOHC_tar_z_glob[model][experiment] = da.nansum(fOHC_cells_tar_z[model][experiment], axis=(1,2,3)).compute()  # (nt,)

#%% 8. Figures

#%%% 8.1 Timeseries
# fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
# for model in Ensemble:
#     ax.plot(np.arange(len(OHC_tar_z_glob[model])), OHC_tar_z_glob[model] / 1e21, label = model, lw=1.5)
# ax.set_title(f"Global OHC ({Depth_upp}-{Depth_low} m)", fontsize=14)
# ax.set_xlabel("Experiment model years")
# ax.set_ylabel("OHC (ZJ)")
# ax.set_ylim(0, 4000)
# ax.legend()
# ax.grid(True, alpha=0.3)
# plt.tight_layout()

#%%

from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
# --- 1. Define model colors manually (adjust as you wish)
model_colors = {
    'CCSM4': '#7100A8',
    'CCSM4-Utrecht': '#FA1EE0',
    'CESM1.2': '#52FAFA',
    'CESM2': '#000000',
    'COSMOS': '#6B4F1D',
    'EC-Earth3-LR': '#1E38FA',
    'GISS-E2-1-G': '#15D11A',
    'HadCM3': '#FF9E63',
    'HadGEM3': '#F52731',
    'IPSL-CM6A-LR': '#8A8A8A',}
# --- 2. Marker styles ---
hist_marker = 'x'  
markers = {'ssp126': 'o', 'ssp245': 's', 'ssp585': '^'}  # future experiments
# --- 3. Plot historical ensemble ---
for model in Ensemble:
    color = model_colors.get(model, 'gray')
    y = OHC_tar_z_glob[model] / 1e21
    x = np.arange(len(y)) + 1850
    ax.plot(x, y, color=color, lw=1.8)
    # Add sparse markers for historical
    step = 25
    ax.plot(x[::step], y[::step], linestyle='none', marker=hist_marker, color=color, markersize=6)
# --- 4. Plot future ensemble ---
for model in Ensemblef:
    color = model_colors.get(model, 'gray')
    for experiment in ['ssp126', 'ssp245', 'ssp585']:
        y = fOHC_tar_z_glob[model][experiment] / 1e21
        x = np.arange(len(y)) + 1850
        ax.plot(x, y, color=color, lw=1.8)
        step = 25
        ax.plot(x[::step], y[::step], linestyle='none', marker=markers[experiment], color=color, markersize=5.5)
# --- 5. Build clean legend ---
# (a) Models — color lines
model_handles = [
    Line2D([0], [0], color=color, lw=2, label=model)
    for model, color in model_colors.items() if model in Ensemble or model in Ensemblef]
# (b) Experiments — markers
exp_handles = [Line2D([0], [0], color='k', marker=hist_marker, linestyle='none', markersize=6, label='midPliocene')
] + [
    Line2D([0], [0], color='k', marker=markers[exp], linestyle='none', markersize=6, label=exp)
    for exp in markers]
# Combine legends (two separate sections)
first_legend = ax.legend(handles=model_handles, title="Models", bbox_to_anchor=(0.26, 1.0), fontsize=12)
ax.add_artist(first_legend)
ax.legend(handles=exp_handles, title="Experiments", bbox_to_anchor=(0.475, 1.0), fontsize=12)
# --- 6. Formatting ---
#ax.set_title(f"Global OHC ({Depth_upp}-{Depth_low} m)", fontsize=14)
ax.set_xlabel("Model year", fontsize = 14)
ax.set_ylabel("OHC (ZJ)", fontsize = 14)
ax.tick_params(axis = 'both', labelsize = 14)
ax.set_ylim(-1000, 7000)
ax.grid(True, alpha=0.3)
#plt.savefig(f"FigureS3.png", dpi=300, bbox_inches='tight')
#plt.savefig(f"FigureS3.pdf", dpi=600, bbox_inches='tight')
plt.show()



