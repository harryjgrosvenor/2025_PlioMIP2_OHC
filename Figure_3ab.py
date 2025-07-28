# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 11:41:40 2025

@author: Harry J Grosvenor
"""
#%%% 1. Import required packages

import numpy as np
import regionmask
import xarray as xa
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import os
import cartopy
from matplotlib.lines import Line2D

#%% 2. Define Functions:
    
def find_variable(ds, variable_names):
    """
    Find a variable in a dataset given a list of possible variable names.    
    Parameters:
        ds (xarray.Dataset): Dataset to search for the variable.
        variable_names (list): List of possible variable names.       
    Returns:
        variable: Variable object if found, None otherwise.
    """
    for name in variable_names:
       if name in ds.variables:
           return ds[name]
    return None

#%% 3. Read in files

# Define the experiments investigated
# E280 = piControl
# Eoi400 = midPliocene  
Experiments = ["E280", "Eoi400"]

# Read in potential temperature files
# Define the Ensemble of models used for temp
Ensemble_t = ["CCSM4", "CCSM4-Utrecht", "CESM1.2", "CESM2", "COSMOS",
              "EC-Earth3-LR", "GISS-E2-1-G", "HadCM3", "HadGEM3", "IPSL-CM6A-LR",
              "MIROC4M", "NorESM-L", "NorESM1-F"]

# Initialize a dictionary to store datasets for each experiment
datasets_t = {model: {} for model in Ensemble_t}

# Loop through each experiment
for model in Ensemble_t:
    # Loop through each model within the experiment
    for experiment in Experiments:
        # Construct file path for the current experiment and model
        file_path = f"Data/{experiment}_{model}_thetao.nc"
        if os.path.exists(file_path):
            try:
                # Read in dataset for the current experiment and model
                datasets_t[model][experiment] = xa.open_dataset(file_path)
            except Exception as e:
                print(f"Error reading {experiment} data for {model}: {e}")
        else:
            pass
        if not datasets_t[model]:
            del datasets_t[model]
        
# Salinity read in:
Ensemble_s = ["CCSM4", "CCSM4-Utrecht", "CESM1.2", "CESM2", "COSMOS",
              "EC-Earth3-LR", "GISS-E2-1-G", "HadCM3", "HadGEM3", "IPSL-CM6A-LR",
              "MIROC4M", "NorESM-L", "NorESM1-F"]

# Initialize a dictionary to store datasets for each experiment
datasets_s = {model: {} for model in Ensemble_s}

# Loop through each experiment
for model in Ensemble_s:
    # Loop through each model within the experiment
    for experiment in Experiments:
        # Construct file path for the current experiment and model
        file_path = f"Data/{experiment}_{model}_so.nc"
        if os.path.exists(file_path):
            try:
                # Read in dataset for the current experiment and model
                datasets_s[model][experiment] = xa.open_dataset(file_path)
            except Exception as e:
                print(f"Error reading {experiment} data for {model}: {e}")
        else:
            pass
        if not datasets_s[model]:
            del datasets_s[model]
            
#%% 4. Variables

# List all the possible naming conventions for each variables required
Var_names = {
    'thetao': ['thetao', 'temp_1', 'temp', 'temperature', 'TEMP'],
    'salinity': ['so', 'SALT', 'Salinity', 'sal'],
    'lat': ['lat', 'latitude', 'TLAT', 'nav_lat'],
    'lon': ['lon', 'longitude', 'TLONG', 'nav_lon'],
    'depth': ['depth', 'plev', 'lev','level', 'z_t', 'olevel', 'depth_std', 'deptht', 'depth_1'],
    }


# Initialise Thetas
# Initialise empty Thetas dictionary to store potential temperature data
Thetas = {model: {} for model in Ensemble_t}
Lats_t = {model: {} for model in Ensemble_t}
Lons_t = {model: {} for model in Ensemble_t}
Depths_t = {model: {} for model in Ensemble_t}

# Assign the correct variable name using our pre-defined function
for model in Ensemble_t:
    for experiment in Experiments:
        # Define thetao variable for model experiments
        Thetas[model][experiment] = datasets_t[model][experiment].thetao
        # Eliminate unnecessary dimension of size 1 in Theta
        Thetas[model][experiment] = np.squeeze(Thetas[model][experiment])
        
# Lat, Lon and Depth will all have the same grid structure between experiments so only need one array:        
for model in Ensemble_t:
    for experiment in Experiments:
        Lats_t[model][experiment] = datasets_t[model][experiment].lat
        Lons_t[model][experiment] = datasets_t[model][experiment].lon
        Depths_t[model]= datasets_t[model]['Eoi400'].depth
        
# Initialise Salinities (some of them are frustratingly at different data management stages to Theta so need
# separate 3d grid for all.)
Sals = {model: {} for model in Ensemble_s}
Lats_s = {model: {} for model in Ensemble_s}
Lons_s = {model: {} for model in Ensemble_s}
Depths_s = {model: {} for model in Ensemble_s}

# Assign the correct variable name using our pre-defined function
for model in Ensemble_s:
    for experiment in Experiments:
        # Define salinity variable for model experiments
        Sals[model][experiment] = find_variable(datasets_s[model][experiment], Var_names['salinity'])
        # Eliminate unnecessary dimension of size 1 in Sals
        Sals[model][experiment] = np.squeeze(Sals[model][experiment])

# Lat, Lon and Depth will all have the same grid structure between experiments so only need one array:        
for model in Ensemble_s:
    for experiment in Experiments:
        Lats_s[model][experiment] = find_variable(datasets_s[model][experiment], Var_names['lat'])
        Lons_s[model][experiment] = find_variable(datasets_s[model][experiment], Var_names['lon'])
        Depths_s[model]= find_variable(datasets_s[model]['E280'], Var_names['depth'])

# Areacello work
# Initialise empty areacello dictionary
areacello = {model: {} for model in Ensemble_t}

# Loop through each model
for model in Ensemble_t:
    if Lats_t[model]['E280'].ndim == 2 and Lons_t[model]['E280'].ndim == 2:
        for experiment in Experiments:
            # Check if both Lats and Lons have 2D dimensions for the current model
            if model in ["CCSM4", "CESM1.2", "CESM2"]:
                # Use the specific file for these models and experiment
                areacello_file = "Data/10.areacello/areacello_CESM2_midPliocene.nc"
                areacello_ds = xa.open_dataset(areacello_file)
                areacello[model][experiment] = areacello_ds.areacello.values
            else:
                areacello_File = f"Data/10.areacello/areacello_{model}_piControl.nc"
                # Open the dataset
                areacello_ds = xa.open_dataset(areacello_File)
                # Add the area cell values to the dictionary for the current model
                areacello[model][experiment] = areacello_ds.areacello.values

#%% 5. Dimensions

# Define lists of possible dimension names for each dimension index
possible_dim_names = {
    0: ['lat', 'latitude', 'nlat', 'nav_lat', 'j', 'y'],
    1: ['lon', 'longitude', 'nlon', 'nav_lon', 'i', 'x'],
    2: ['depth', 'lev','plev','level', 'olevel', 'z_t', 'depth_std', 'deptht', 'depth_1']
    }

# Thetas Dimensions
dim_names_t = {model: {experiment: {} for experiment in Experiments} for model in Ensemble_t}

# Initialize variables to hold the dimension names
for model in Ensemble_t:
    for experiment in Experiments:
        dim_names_t[model][experiment] = {}
        for dim_index in range(3):
            dim_name = next((name for name in possible_dim_names[dim_index] if name in datasets_t[model][experiment].dims), None)
            if dim_name:
                dim_names_t[model][experiment][f'dim_name_{dim_index}'] = dim_name
            else:
                raise ValueError(f"Dimension {dim_index} not found in the {experiment} dataset of {model}.")

Depths_m = {model: {} for model in Ensemble_t}

# Some models have the depth variable measured in centimeters, convert to meters.
for model in Ensemble_t:
    try:
        if Depths_t[model].units =='centimeters' or Depths_t[model].units == 'cm':
            Depths_m[model] = Depths_t[model] / 100
        else:
            Depths_m[model] = Depths_t[model]
        Depths_t[model] = Depths_m[model]
    except:
        pass

# Rearrange dimensions for ease of use later
for model in Ensemble_t:
    for experiment in Experiments:
        Thetas[model][experiment] = Thetas[model][experiment].transpose(
            dim_names_t[model][experiment]['dim_name_0'],
            dim_names_t[model][experiment]['dim_name_1'],
            dim_names_t[model][experiment]['dim_name_2']
        )
        
# Salinities dimensions
dim_names_s = {model: {experiment: {} for experiment in Experiments} for model in Ensemble_t}

# Initialize variables to hold the dimension names
for model in Ensemble_s:
    for experiment in Experiments:
        dim_names_s[model][experiment] = {}
        for dim_index in range(3):
            dim_name = next((name for name in possible_dim_names[dim_index] if name in datasets_s[model][experiment].dims), None)
            if dim_name:
                dim_names_s[model][experiment][f'dim_name_{dim_index}'] = dim_name
            else:
                raise ValueError(f"Dimension {dim_index} not found in the {experiment} dataset of {model}.")

# Some models have the depth variable measured in centimeters, convert to meters.
for model in Ensemble_t:
    try:
        if Depths_s[model].units =='centimeters' or Depths_s[model].units == 'cm':
            Depths_m[model] = Depths_s[model] / 100
        else:
            Depths_m[model] = Depths_s[model]
        Depths_s[model] = Depths_m[model]
    except:
        pass

# Rearrange dimensions for ease of use later
for model in Ensemble_s:
    for experiment in Experiments:
        Sals[model][experiment] = Sals[model][experiment].transpose(
            dim_names_s[model][experiment]['dim_name_0'],
            dim_names_s[model][experiment]['dim_name_1'],
            dim_names_s[model][experiment]['dim_name_2']
        )
        
#%% 6. Constants

R_earth = 6.371e6 # Earth's radius in meters  

#%% 7. Calculate grid cell area for each model
# First big processing batch

# For Potential Temperature
# Depth is always 1D so can be defined outside the if loop.
Depth_ext_t = {}
Depth_edges_t = {}
dDepth_t = {}
# Define Depth_edges for each experiment
for model in Ensemble_t:
    Depth_ext_t[model] = np.concatenate((Depths_t[model], [Depths_t[model][-1] + (Depths_t[model][-1] - Depths_t[model][-2])]))
    Depth_edges_t[model] = np.concatenate(([0], ((Depth_ext_t[model][:-1] + Depth_ext_t[model][1:]) / 2)))
    dDepth_t[model] = Depth_edges_t[model][1:] - Depth_edges_t[model][:-1]

grid_cell_area = {model: {experiment: None for experiment in Experiments} for model in Ensemble_t}

# Then calculate grid cell areas:
for model in Ensemble_t:
    for experiment in Experiments:
        if Lats_t[model][experiment].ndim == 1 and Lons_t[model][experiment].ndim == 1:
            # Regular grid option
            Lat_ext = np.concatenate(([np.max(Lats_t[model][experiment])-180], Lats_t[model][experiment], [np.min(Lats_t[model][experiment])+180]))
            Lat_edges = (Lat_ext[:-1]+ Lat_ext[1:])/2
            # Models differ in how they represent Longitude, so explicitly work with the 
            # maximum and minimum Longitude values.
            Lon_ext = np.concatenate(([np.max(Lons_t[model][experiment])-360], Lons_t[model][experiment], [np.min(Lons_t[model][experiment])+360]))
            Lon_edges = (Lon_ext[:-1]+ Lon_ext[1:])/2
            grid_cell_area[model][experiment] = np.zeros((len(Lats_t[model][experiment]), len(Lons_t[model][experiment])))
            for lat_idx in range(len(Lats_t[model][experiment])):
                for lon_idx in range(len(Lons_t[model][experiment])):
                    lat_spacing = np.abs(Lat_edges[lat_idx + 1] - Lat_edges[lat_idx]) * (np.pi / 180) * R_earth
                    lon_spacing = np.abs(Lon_edges[lon_idx + 1] - Lon_edges[lon_idx]) * (np.pi / 180) * R_earth * np.cos(np.deg2rad(Lats_t[model][experiment][lat_idx]))
                    grid_cell_area[model][experiment][lat_idx, lon_idx] = lat_spacing * lon_spacing
            print(f"Grid cell areas calculated for model {model}: {experiment}")
        elif Lats_t[model][experiment].ndim == 2 and Lons_t[model][experiment].ndim == 2:
            # Irregular grid option
            grid_cell_area[model][experiment] = areacello[model][experiment]
            print(f"Grid cell areas calculated for model {model}: {experiment}")
            
#%% 8. Masks 

Locale_key = 'Arb'

# Define a dictionary for the locales
Locale = {
    'Arb':{
        'long': 'Arabian Sea',
        'short': 'Arb',
        'code': 53
    }
    }

# Mask for Arabian Sea
def mask_data_for_arabian_sea(grid_cell_area, Lons, Lats):
    ar6_reg = Locale[Locale_key]['code']
    # Load AR6 region mask for Arabian Sea
    arabian_sea_mask = regionmask.defined_regions.ar6.all.mask(Lons,Lats)
    arabian_sea_mask = arabian_sea_mask == ar6_reg
    # Apply the mask to the data
    masked_grid_cell_area = grid_cell_area * arabian_sea_mask
    return masked_grid_cell_area

# For Potential Temperature
# Initialise dictionaries for Arabian Sea Weights and for storing the potential temperature
Arb_Sea_grid_cell_area = {model: {} for model in Ensemble_t}

# Iterate through each model
for model in Ensemble_t:
    for experiment in Experiments:
        lat = Lats_t[model][experiment]
        lon = Lons_t[model][experiment]
        gca = grid_cell_area[model][experiment]
        # Mask data for Arabian Sea
        Arb_Sea_grid_cell_area[model][experiment] = mask_data_for_arabian_sea(gca, lon, lat)

#%% 9. Calculate weighted-average depth profile


# Temperature Profiles
# Initialize Arb_Sea_profiles dictionary
Arb_Sea_t_profiles = {model: {experiment: {} for experiment in Experiments} for model in Ensemble_t}

# Loop through each model
for model in Ensemble_t:
    # Loop through each experiment for the current model
    for experiment in Experiments:
        # Get the masked grid cell area for the current model and experiment
        masked_gca = Arb_Sea_grid_cell_area[model][experiment]   
        # Initialize arrays to store the weighted sum of temperatures and the total weighted area
        weighted_sum = np.zeros_like(Thetas[model][experiment][0, 0, :])
        total_weighted_area = np.zeros_like(Thetas[model][experiment][0, 0, :])
        # Loop through each depth level
        for depth in range(Thetas[model][experiment].shape[2]):
            # Create a mask for cells that have a temperature value at the current depth
            valid_mask = ~np.isnan(Thetas[model][experiment][:, :, depth])
            # Calculate the weighted sum and the total weighted area only for valid cells
            weighted_sum[depth] = np.sum(Thetas[model][experiment][:, :, depth] * masked_gca * valid_mask)
            total_weighted_area[depth] = np.sum(masked_gca * valid_mask)
        # Compute the weighted average depth profile
        weighted_avg_t_profile = np.divide(weighted_sum, total_weighted_area, out=np.full_like(weighted_sum, np.nan), where=total_weighted_area!=0)
        # Store the weighted average depth profile in Arb_Sea_profiles
        Arb_Sea_t_profiles[model][experiment] = weighted_avg_t_profile
        
# Salinity Profiles
# Initialize Arb_Sea_profiles dictionary
Arb_Sea_s_profiles = {model: {experiment: {} for experiment in Experiments} for model in Ensemble_s}

# Loop through each model
for model in Ensemble_s:
    # Loop through each experiment for the current model
    for experiment in Experiments:
        # Get the masked grid cell area for the current model and experiment
        masked_gca = Arb_Sea_grid_cell_area[model][experiment]
        # Initialize arrays to store the weighted sum of salinities and the total weighted area
        weighted_sum = np.zeros(Sals[model][experiment].shape[2])
        total_weighted_area = np.zeros(Sals[model][experiment].shape[2])
        # Loop through each depth level
        for depth in range(Sals[model][experiment].shape[2]):
            # Create a mask for cells that have a salinity value at the current depth
            valid_mask = ~np.isnan(Sals[model][experiment][:, :, depth])
            valid_masked_gca = masked_gca * valid_mask
            # Calculate the weighted sum and the total weighted area only for valid cells
            weighted_sum[depth] = np.nansum(Sals[model][experiment][:, :, depth] * valid_masked_gca)
            total_weighted_area[depth] = np.nansum(valid_masked_gca)
        # Compute the weighted average depth profile
        weighted_avg_s_profile = np.divide(weighted_sum, total_weighted_area, out=np.full_like(weighted_sum, np.nan), where=total_weighted_area != 0)
        # Store the weighted average depth profile in Arb_Sea_s_profiles
        Arb_Sea_s_profiles[model][experiment] = weighted_avg_s_profile
        
#%% 10. Interpolate profiles with respect to depth onto standard depths

# Define the standard depths
standard_depths = []
# Add depths from 0 to 100 every 5 meters
standard_depths.extend(np.arange(5, 200, 5))
# Add depths from 100 to 250 every 10 meters
standard_depths.extend(np.arange(200, 401, 10))
# Add depths from 250 to 500 every 25 meters
standard_depths.extend(np.arange(401, 751, 25))
# Add depths from 500 to 1000 every 50 meters
standard_depths.extend(np.arange(751, 1001, 50))
# Add depths from 1000 to 2000 every 100 meters
standard_depths.extend(np.arange(1000, 2001, 100))
# Add depths from 2000 to 6000 every 250 meters
standard_depths.extend(np.arange(2000, 6001, 250))

# Convert to numpy array
standard_depths = np.array(standard_depths)

# Interpolate Temperature profiles on to standard depths
Arb_Sea_t_profiles_int = {model: {experiment: {} for experiment in Experiments} for model in Ensemble_t}

# Interpolate each model's profiles onto the standard depths
for model in Ensemble_t:
    for experiment in Experiments:
        # Extract depth profile and corresponding temperatures
        depths = Depths_t[model]
        temps = Arb_Sea_t_profiles[model][experiment]     
        # Perform interpolation
        t_interp = scipy.interpolate.interp1d(depths, temps, kind='linear', bounds_error=False, fill_value=np.nan)       
        # Interpolate temperatures at standard depths
        interpolated_temps = t_interp(standard_depths)       
        # Store interpolated temperatures and depths
        Arb_Sea_t_profiles_int[model][experiment]['depths'] = standard_depths
        Arb_Sea_t_profiles_int[model][experiment]['temperatures'] = interpolated_temps
        
# Interpolate Salinity profiles on to standard depths
Arb_Sea_s_profiles_int = {model: {experiment: {} for experiment in Experiments} for model in Ensemble_t}

# Interpolate each model's profiles onto the standard depths
for model in Ensemble_t:
    for experiment in Experiments:
        # Extract depth profile and corresponding temperatures
        depths = Depths_s[model]
        sals = Arb_Sea_s_profiles[model][experiment]     
        # Perform interpolation
        s_interp = scipy.interpolate.interp1d(depths, sals, kind='linear', bounds_error=False, fill_value=np.nan)       
        # Interpolate temperatures at standard depths
        interpolated_sals = s_interp(standard_depths)       
        # Store interpolated temperatures and depths
        Arb_Sea_s_profiles_int[model][experiment]['depths'] = standard_depths
        Arb_Sea_s_profiles_int[model][experiment]['salinities'] = interpolated_sals
        
#%% 11. Generate Ensemble mean

# Temperature ensemble mean
all_temps = {experiment: [] for experiment in Experiments}
Arb_Sea_Ensemble_t_mn = {experiment: {} for experiment in Experiments}

models_exc_t = {'CCSM4-Utrecht', 'NorESM-L'}

Ensemble_t_filt = list(set(Ensemble_t) - models_exc_t)

# Collect temperatures across all models and experiments
for experiment in Experiments:
    for model in Ensemble_t_filt:
            temperatures = Arb_Sea_t_profiles_int[model][experiment]['temperatures']
            all_temps[experiment].append(temperatures)
        # Convert to numpy array for easier nanmean calculation
    all_temps[experiment] = np.array(all_temps[experiment])
    # Calculate mean temperatures ignoring NaNs
    Arb_Sea_Ensemble_t_mn[experiment] = np.nanmean(all_temps[experiment], axis=0)

# Salinity ensemble mean
all_sals = {experiment: [] for experiment in Experiments}
Arb_Sea_Ensemble_s_mn = {experiment: {} for experiment in Experiments}

models_exc_s = {'NorESM-L'}

Ensemble_s_filt = list(set(Ensemble_s) - models_exc_s)

# Collect salinities across all models and experiments
for experiment in Experiments:
    for model in Ensemble_s_filt:
            salinities = Arb_Sea_s_profiles_int[model][experiment]['salinities']
            all_sals[experiment].append(salinities)
        # Convert to numpy array for easier nanmean calculation
    all_sals[experiment] = np.array(all_sals[experiment])
    # Calculate mean temperatures ignoring NaNs
    Arb_Sea_Ensemble_s_mn[experiment] = np.nanmean(all_sals[experiment], axis=0)
    
#%% 12. Thermocline depth

# Thermocline location
Thermocline_depths = {model: {experiment:() for experiment in Experiments} for model in Ensemble_t}
Thermocline_z_idx = {model: {experiment:() for experiment in Experiments} for model in Ensemble_t}

# Locate thermocline in every model
for model in Ensemble_t:
    for experiment in Experiments:
        profile = Arb_Sea_t_profiles[model][experiment]
        z_level = Depths_t[model]
        # Remove NaNs which represent the seafloor
        valid_mask = ~np.isnan(profile)
        valid_profile = profile[valid_mask]
        valid_z_level = z_level[valid_mask]
        # Calculate the gradient of the temperature profile
        dTdz = np.gradient(valid_profile, valid_z_level)
        # Find the depth where the gradient is maximum
        if len(dTdz) > 0:
            thermocline_depth = valid_z_level[np.argmax(np.abs(dTdz))]
            thermocline_depth_idx = np.argmax(np.abs(dTdz))
        else:
            thermocline_depth = np.nan
            thermocline_depth_idx = np.nan
        Thermocline_depths[model][experiment] = thermocline_depth.values
        Thermocline_z_idx[model][experiment] = thermocline_depth_idx
        
# Calculate the thermocline depth for the Ensemble_t mean in each experiment
Thermocline_Ensemble_t_mn = {experiment: [] for experiment in Experiments}     
Thermocline_Ensemble_t_idx = {experiment: [] for experiment in Experiments}   
      
for experiment in Experiments:
    Thermocline_depths_array = []
    for model in Ensemble_t:
        thermocline_depth = Thermocline_depths[model][experiment]
        Thermocline_depths_array.append(thermocline_depth)
    Thermocline_Ensemble_t_mn[experiment] = np.mean(Thermocline_depths_array) if len (Thermocline_depths_array) > 0 else np.nan
    Thermocline_Ensemble_t_idx[experiment] = np.argmin(np.abs(standard_depths - Thermocline_Ensemble_t_mn[experiment]))
    
    
#%% 14. Clusters: Land-Sea Mask

# Land-sea masks
LS_mask = {"CCSM4": "midPliocene", "CCSM4-Utrecht": "midPliocene", "CESM1.2": "midPliocene", "CESM2": "midPliocene", "COSMOS": "midPliocene",
           "EC-Earth3-LR": "midPliocene", "GISS-E2-1-G": "midPliocene", "HadCM3": "midPliocene", "HadGEM3": "piControl", "IPSL-CM6A-LR": "piControl",
           "MIROC4M": "midPliocene", "NorESM-L": "piControl", "NorESM1-F": "piControl"}

# Define colors
browns = [
    "#8B4513", "#A0522D", "#D2691E", "#DEB887", "#F4A460",
    "#CD853F", "#8B4513", "#A0522D", "#D2691E", "#DEB887", "#F4A460"
]

#pliocene_colors = plt.cm.Browns(np.linspace(0.4, 1, list(LS_mask.values()).count("Pliocene")))
piControl_colors = plt.cm.Blues(np.linspace(0.4, 1, list(LS_mask.values()).count("piControl")))

# Assign colors to each model
color_ass = {}
pliocene_index = 0
piControl_index = 0

for model in Ensemble_t:
    if LS_mask[model] == "midPliocene":
        color_ass[model] = browns[pliocene_index % len(browns)]
        pliocene_index += 1
    else:
        color_ass[model] = piControl_colors[piControl_index]
        piControl_index += 1

# Function to get color from colormap based on land-sea mask
def get_lsm_color(model):
    return (color_ass[model])

Plio_LS_models = [model for model in Ensemble_t_filt if LS_mask[model] == "midPliocene"]
piC_LS_models = [model for model in Ensemble_t_filt if LS_mask[model] == "piControl"]

# Temperature profile work for LSM clustering
Plio_mask_temps =  {experiment: [] for experiment in Experiments}
PiC_mask_temps = {experiment: [] for experiment in Experiments}
Plio_mask_t_mns = {experiment: [] for experiment in Experiments}
PiC_mask_t_mns = {experiment: [] for experiment in Experiments}

for experiment in Experiments:
    for model in Ensemble_t:
        temperatures = Arb_Sea_t_profiles_int[model][experiment]['temperatures']
        if model in Plio_LS_models:
            Plio_mask_temps[experiment].append(temperatures)
        elif model in piC_LS_models:
            PiC_mask_temps[experiment].append(temperatures)
        else:
            pass
    Plio_mask_temps[experiment] = np.array(Plio_mask_temps[experiment])
    PiC_mask_temps[experiment] = np.array(PiC_mask_temps[experiment])
    Plio_mask_t_mns[experiment] = np.nanmean(Plio_mask_temps[experiment], axis = 0)
    PiC_mask_t_mns[experiment] = np.nanmean(PiC_mask_temps[experiment], axis = 0)

# Thermocline depths for mean profiles:
Thermoclines_plio = {experiment: [] for experiment in Experiments}
Thermoclines_piC = {experiment: [] for experiment in Experiments}

for experiment in Experiments:
    for model in Ensemble_t:
        if model in Plio_LS_models:
            Thermoclines_plio[experiment].append(Thermocline_depths[model][experiment])
        if model in piC_LS_models:
            Thermoclines_piC[experiment].append(Thermocline_depths[model][experiment])
            
Thermocline_plio_mn = {experiment: np.mean(depths) if len(depths) > 0 else np.nan for experiment, depths in Thermoclines_plio.items()}
Thermocline_piC_mn = {experiment: np.mean(depths) if len(depths) > 0 else np.nan for experiment, depths in Thermoclines_piC.items()}


# Find standard_depth index that correlates closest to thermocline mean depth
Thermocline_plio_idx = {experiment: [] for experiment in Experiments}
Thermocline_piC_idx = {experiment: [] for experiment in Experiments}

for experiment in Experiments:
    Thermocline_plio_idx[experiment] = np.argmin(np.abs(standard_depths - Thermocline_plio_mn[experiment]))
    Thermocline_piC_idx[experiment] = np.argmin(np.abs(standard_depths - Thermocline_piC_mn[experiment]))
    
# Salinity profile work for LSM clustering
# Temperature profile work for LSM clustering
Plio_mask_sals =  {experiment: [] for experiment in Experiments}
PiC_mask_sals = {experiment: [] for experiment in Experiments}
Plio_mask_s_mns = {experiment: [] for experiment in Experiments}
PiC_mask_s_mns = {experiment: [] for experiment in Experiments}

for experiment in Experiments:
    for model in Ensemble_s:
        salinities = Arb_Sea_s_profiles_int[model][experiment]['salinities']
        if model in Plio_LS_models:
            Plio_mask_sals[experiment].append(salinities)
        elif model in piC_LS_models:
            PiC_mask_sals[experiment].append(salinities)
        else:
            pass
    Plio_mask_sals[experiment] = np.array(Plio_mask_sals[experiment])
    PiC_mask_sals[experiment] = np.array(PiC_mask_sals[experiment])
    Plio_mask_s_mns[experiment] = np.nanmean(Plio_mask_sals[experiment], axis = 0)
    PiC_mask_s_mns[experiment] = np.nanmean(PiC_mask_sals[experiment], axis = 0)


#%%% Figures 3ab
# for land-sea mask
Fig3ab, axs = plt.subplots(1, 2, figsize=(32, 16), dpi=300)
for model in Ensemble_t_filt:
    depths = Depths_t[model]
    differences = Arb_Sea_t_profiles[model]['Eoi400'] - Arb_Sea_t_profiles[model]['E280']
    color = get_lsm_color(model)
    axs[0].plot(differences, depths, label = f"{model}", color = color, linewidth = 1, alpha = 0.4)
    axs[0].scatter(differences, depths, color=color, marker='x', s=40, alpha = 0.4)
    #axs.scatter(differences[Thermocline_z_idx[model]['Eoi400']], depths[Thermocline_z_idx[model]['Eoi400']], color = 'red', marker = '*', s = 100, edgecolor = 'k', zorder = 6)
    #axs.scatter(differences[Thermocline_z_idx[model]['E280']], depths[Thermocline_z_idx[model]['E280']], color = 'yellow', marker = 'o', s = 100, edgecolor = 'k', zorder = 5)
# Define difference variables for the means taken
Ensemble_t_mn_diff = Arb_Sea_Ensemble_t_mn['Eoi400'] - Arb_Sea_Ensemble_t_mn['E280']
plio_mask_diff = Plio_mask_t_mns['Eoi400'] - Plio_mask_t_mns['E280']
pic_mask_diff = PiC_mask_t_mns['Eoi400'] - PiC_mask_t_mns['E280']
# Plot the respective mean variables
axs[0].scatter(plio_mask_diff[Thermocline_plio_idx['E280']], standard_depths[Thermocline_plio_idx['E280']], color = 'yellow', marker = '*', s = 800, edgecolor = 'k', zorder = 10)
axs[0].plot(Ensemble_t_mn_diff, standard_depths, label = 'Ensemble Mean', color = 'black', linewidth = 5)
axs[0].plot(pic_mask_diff, standard_depths, label = "$\mathit{piControl}$ MC mean", color = "blue", linewidth = 5)
axs[0].plot(plio_mask_diff, standard_depths, label = "$\mathit{midPliocene}$ MC mean", color = "brown", linewidth = 5)
axs[0].scatter(plio_mask_diff[Thermocline_plio_idx['Eoi400']], standard_depths[Thermocline_plio_idx['Eoi400']], color = 'red', marker = '*', s = 800, edgecolor = 'k', zorder = 10)
axs[0].scatter(pic_mask_diff[Thermocline_piC_idx['E280']], standard_depths[Thermocline_piC_idx['E280']], color = 'yellow', marker = '*', s = 800, edgecolor = 'k', zorder = 10, label = "$\mathit{piControl}$ therm.")
axs[0].scatter(pic_mask_diff[Thermocline_piC_idx['Eoi400']], standard_depths[Thermocline_piC_idx['Eoi400']], color = 'red', marker = '*', s = 800, edgecolor = 'k', zorder = 10, label = "$\mathit{midPliocene}$ therm.")
axs[0].scatter(Ensemble_t_mn_diff[Thermocline_Ensemble_t_idx['E280']], standard_depths[Thermocline_Ensemble_t_idx['E280']], color = 'yellow', marker = '*', s = 800, edgecolor = 'k', zorder =10)
axs[0].scatter(Ensemble_t_mn_diff[Thermocline_Ensemble_t_idx['Eoi400']], standard_depths[Thermocline_Ensemble_t_idx['Eoi400']], color = 'red', marker = '*', s = 800, edgecolor = 'k', zorder =10)
axs[0].set_xlabel(r"Δ $\theta$ (°C)", fontsize = 24)
axs[0].set_ylabel("Depth (m)", fontsize = 24)
#axs.set_yscale('log')
#axs.set_title("Δ Temperature-depth profile", fontsize = 16)
axs[0].set_xlim(-3.5, 4.5)
axs[0].set_ylim(0, 1500)
axs[0].tick_params(axis='both', labelsize=24)
axs[0].invert_yaxis()
#axs.set_yscale('log')  # This will make the y-axis logarithmic
# Extract handles and labels from subplot 0
handles, labels = axs[0].get_legend_handles_labels()
axs[0].axvline(x=0, color='black', linewidth=1.5, linestyle='--')  
axs[0].grid(True)
for model in Ensemble_t_filt:
    depths = Depths_t[model]
    differences = Arb_Sea_s_profiles[model]['Eoi400'] - Arb_Sea_s_profiles[model]['E280']
    color = get_lsm_color(model)
    axs[1].plot(differences, depths, label = f"{model}", color = color, linewidth = 1, alpha = 0.4)
    axs[1].scatter(differences, depths, color=color, marker='x', s=40, alpha = 0.4)
# Define difference variables for the means taken
Ensemble_s_mn_diff = Arb_Sea_Ensemble_s_mn['Eoi400'] - Arb_Sea_Ensemble_s_mn['E280']
plio_mask_diff = Plio_mask_s_mns['Eoi400'] - Plio_mask_s_mns['E280']
pic_mask_diff = PiC_mask_s_mns['Eoi400'] - PiC_mask_s_mns['E280']
# Plot the respective mean variables
axs[1].plot(pic_mask_diff, standard_depths, label = "$\mathit{piControl}$ MC mean", color = "blue", linewidth = 5)
axs[1].plot(plio_mask_diff, standard_depths, label = "$\mathit{midPliocene}$ MC mean", color = "brown", linewidth = 5)
axs[1].plot(Ensemble_s_mn_diff, standard_depths, label = 'Ensemble Mean', color = 'black', linewidth = 5)
axs[1].set_xlabel(r"Δ S (%$_{00}$)", fontsize = 24)
axs[1].set_xlim(-3, 1)
axs[1].set_ylim(0, 1500)
axs[1].tick_params(axis='both', labelsize=24)
axs[1].invert_yaxis()
#axs.set_yscale('log')  # This will make the y-axis logarithmic
axs[1].legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 0.0), fontsize=20)
axs[1].axvline(x=0, color='black', linewidth=1.5, linestyle='--')  
axs[1].grid(True)
#Fig3ab.suptitle(f"{Locale[Locale_key]['long']}: Land Sea Mask comparison")
#plt.savefig(f"....png", dpi=600, bbox_inches='tight')
#plt.savefig(f"....pdf", dpi=600, bbox_inches='tight')
plt.show()