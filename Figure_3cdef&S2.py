# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:21:42 2025

@author: Harry J Grosvenor
"""
#%% 1. Import packages

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xa
import pandas
import regionmask
import cartopy
import cmocean

#%% 2. Universal Pre-processing

# Define the list of experiments
# Experiment nomenclature:
# E280 = piControl
# Eoi400 = midPliocene
Experiments = ["E280", "Eoi400"]

# Define months for plotting
Months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 

Seasons = ['MAM', 'JJA', 'SON', 'DJF']
# Season indexes
spr_idx = [2, 3, 4]
sum_idx = [5, 6, 7]
aut_idx = [8, 9, 10]
win_idx = [11, 0, 1]

# Thresholds for statistics
Threshold_frac = 2/3

# Outline ensembles
# OHC list of models
Ensemble_OHC = ["CCSM4", "CCSM4-Utrecht","CESM1.2", "CESM2", "COSMOS",
                "EC-Earth3-LR", "GISS-E2-1-G", "HadCM3", "HadGEM3", "IPSL-CM6A-LR"]

# Zonal and meridional wind components list of models      
Ensemble_uva = ["CCSM4", "CESM1.2", "CESM2", "COSMOS",
               "EC-Earth3-LR", "GISS-E2-1-G", "HadCM3", "HadGEM3", "IPSL-CM6A-LR",
               "MIROC4M", "NorESM1-F"]

Lat = np.arange(-89.5, 90, 1)
Lon = np.arange(0, 360, 1)
Time = np.arange(1 ,13 , 1)

# Read in the pliocene land-sea mask file
plio_lsm = xa.open_dataset("Data/PRISM4_LSM_v1.0.nc")["p4_lsm"]
# Reindex plio_lsm from -180 to 180 to 0 to 360
lat_size, lon_size = plio_lsm.shape
plio_lsm_reindexed = np.empty_like(plio_lsm)  # Create an empty array with the same shape
# Fill in the new array with data from plio_lsm
plio_lsm_reindexed[:, :lon_size//2] = plio_lsm[:, lon_size//2:]  # 0 to 180
plio_lsm_reindexed[:, lon_size//2:] = plio_lsm[:, :lon_size//2]  # 180 to 360
# Create a mask where plio_lsm_reindexed equals 1
plio_msk = (plio_lsm_reindexed == 1)

# Define piControl land-sea mask usinf regionmask
pi_lsm = regionmask.defined_regions.natural_earth_v5_1_2.land_110.mask(Lon, Lat)
pi_msk = ~pi_lsm.isnull()

# Land-sea mask (LSM) clustering:
# Land-sea masks
LS_mask = {"CCSM4": "Pliocene", "CCSM4-Utrecht": "Pliocene", "CESM1.2": "Pliocene", "CESM2": "Pliocene", "COSMOS": "Pliocene",
           "EC-Earth3-LR": "Pliocene", "GISS-E2-1-G": "Pliocene", "HadCM3": "Pliocene", "HadGEM3": "piControl", "IPSL-CM6A-LR": "piControl",
           "MIROC4M": "Pliocene", "NorESM1-F": "piControl"}

#%% 3. Ocean Heat Content Processing and Statistics

#%%% 3.1 OHC file read in and define variables

datasets_therm = {model: {experiment: {} for experiment in Experiments} for model in Ensemble_OHC}

# Loop through each experiment
for model in Ensemble_OHC:
    # Loop through each model within the experiment
    for experiment in Experiments:
        try:
            # Construct file path for the current experiment and model
            file_path = f"Data/PlioMIP2/Thermoclines/{experiment}_{model}_1deg_thermocline.nc"
            # Read in dataset for the current experiment and model
            datasets_therm[model][experiment] = xa.open_dataset(file_path)
        except:
            print(f"File not found for {model} {experiment}")
            
# OHC Variables
t_depths = {model: {experiment: {} for experiment in Experiments} for model in Ensemble_OHC}

# Define Ocean Heat Content variable for each model and experiment
for model in Ensemble_OHC:
    for experiment in Experiments:
        t_depths[model][experiment] = datasets_therm[model][experiment].depth
        
#%%%% 3.1.1 Apply land-sea mask

# Apply masks depending on experiment
for model in Ensemble_OHC:    
    for experiment in Experiments:
        if experiment == 'Eoi400' and LS_mask[model] == "Pliocene":
            base_mask = plio_msk
        elif experiment == 'Eoi400' and LS_mask[model] == "piControl":
            base_mask = pi_msk
        elif experiment == 'E280':
            base_mask = pi_msk
        else:
            continue  # Skip non-relevant experiments
        # If the mask is a numpy array, convert it to an xarray.DataArray
        if isinstance(base_mask, np.ndarray):
            # Create DataArray with the same coordinates and dimensions as the target DataArray
            mask = xa.DataArray(
                base_mask,
                dims=["lat", "lon"],  # Adjust these names if they differ in your DataArray
                coords={
                    "lat": t_depths[model][experiment].coords["lat"],
                    "lon": t_depths[model][experiment].coords["lon"]
                }
            )
        else:
            mask = base_mask  # It's already an xarray.DataArray
        # Broadcast the mask to 3D to match the DataArray
        mask = mask.broadcast_like(t_depths[model][experiment].isel(time=0))
        # Apply the mask using xarray's .where()
        t_depths[model][experiment] = t_depths[model][experiment].where(~mask, other=np.nan)
        
#%%% 3.2 OHC statistics
#%%%% 3.2.1 Thermocline 

# Thermocline Depth

# Seasonal Data
t_depth_seasons = {model: {experiment: np.zeros((180, 360, 4)) for experiment in Experiments} for model in Ensemble_OHC}
# Seasonal calculation for pr
for model in Ensemble_OHC:
    for experiment in Experiments:
        # Initialize the array with shape (lat, lon, seasons)
        # Assigning the mean values to each season (0: spring, 1: summer, 2: autumn, 3: winter)
        t_depth_seasons[model][experiment][:, :, 0] = np.mean(t_depths[model][experiment][:, :, spr_idx], axis=2)
        t_depth_seasons[model][experiment][:, :, 1] = np.mean(t_depths[model][experiment][:, :, sum_idx], axis=2)
        t_depth_seasons[model][experiment][:, :, 2] = np.mean(t_depths[model][experiment][:, :, aut_idx], axis=2)
        t_depth_seasons[model][experiment][:, :, 3] = np.mean(t_depths[model][experiment][:, :, win_idx], axis=2)
# Inirialise array for differences for each model
t_depth_season_diffs = {model: np.zeros((180, 360, 4)) for model in Ensemble_OHC}
# Run loop to calculate differences in each model
for model in Ensemble_OHC:
    for t in range(len(Seasons)):
        t_depth_season_diffs[model][:,:,t] = t_depth_seasons[model]['Eoi400'][:,:,t] - t_depth_seasons[model]['E280'][:,:,t]

# Seasonal stippling:
# Stippling work:
t_depth_season_diff_array = np.array([t_depth_season_diffs[model].data for model in Ensemble_OHC])
# shape info
models_therm, lat, lon, t = t_depth_season_diff_array.shape
# Define empy arrays for stippling work:
sig_agrees_t_depth_seas = np.zeros((lat, lon, t), dtype = bool)

for timestep in range(t):
    data_diff = t_depth_season_diff_array[:,:,:, timestep]
    # Determine direction of change for each model (True for positive, false for negative, NaN for no change)
    direc_change_t_depth_seas = np.where(np.isnan(data_diff), np.nan, data_diff > 0)
    # Exclude the cases where data stays the same
    direc_change_t_depth_seas = np.where(data_diff == 0, np.nan, direc_change_t_depth_seas)
    # Count number that agree in the positive direction and those in the negative
    agree_no_plus_t_depth_seas = np.nansum(direc_change_t_depth_seas ==1, axis = 0)
    agree_no_neg_t_depth_seas = np.nansum(direc_change_t_depth_seas ==0, axis = 0)
    # Determine threshold for stippling
    threshold_t_depth_seas = Threshold_frac * models_therm
    sig_agree = (agree_no_plus_t_depth_seas >= threshold_t_depth_seas) | (agree_no_neg_t_depth_seas >= threshold_t_depth_seas)
    sig_agrees_t_depth_seas[:,:, timestep] = sig_agree

# Seasonal averages
t_depth_season = {experiment: np.zeros((180, 360, 4)) for experiment in Experiments}

for experiment in Experiments:
    # Stack models' data along a new axis (axis=0) to create a 4D array (models, lat, lon, time)
    t_depths_stack = np.stack([t_depth_seasons[model][experiment] for model in Ensemble_OHC], axis=0)   
    # Compute the mean across models (axis=0), ignoring NaNs, to retain lat, lon, time dimensions
    t_depth_season[experiment] = np.nanmean(t_depths_stack, axis=0)

t_depth_season['Eoi400'] = np.where(~np.expand_dims(plio_msk, axis=-1), t_depth_season['Eoi400'], np.nan)

# Calculate the difference array
t_depth_season_diff = t_depth_season['Eoi400'] - t_depth_season['E280']

#%% 4. Zonal and Meridional wind components processing and statistics

#%%% 4.1 U, V file read in and variable definition

# Initialise wind component datasets
datasets_ua = {model: {experiment: {} for experiment in Experiments} for model in Ensemble_uva}
datasets_va = {model: {experiment: {} for experiment in Experiments} for model in Ensemble_uva}

# Link files to dataset dictionaries
for model in Ensemble_uva:
    for experiment in Experiments:
        try:
            filepath_u = f"Data/{experiment}_{model}_1deg_ua_climatology.nc"
            filepath_v = f"Data/{experiment}_{model}_1deg_va_climatology.nc"
            datasets_ua[model][experiment] = xa.open_dataset(filepath_u)
            datasets_va[model][experiment] = xa.open_dataset(filepath_v)
        except:
            print(f"ua or va file not found for {model} {experiment}")

# Initialise wind component dictionaries
uas = {model: {experiment: {} for experiment in Experiments} for model in Ensemble_uva}
vas = {model: {experiment: {} for experiment in Experiments} for model in Ensemble_uva}

# Define meridional and zonal components for each model and experiment
for model in Ensemble_uva:
    for experiment in Experiments:
        uas[model][experiment] = datasets_ua[model][experiment].ua
        vas[model][experiment] = datasets_va[model][experiment].va

ua_seasons = {model: {experiment: None for experiment in Experiments} for model in Ensemble_uva}
va_seasons = {model: {experiment: None for experiment in Experiments} for model in Ensemble_uva}

# Seasonal variable
for model in Ensemble_uva:
    for experiment in Experiments:
        # Initialize the array with shape (lat, lon, seasons)
        # Assuming Expt_avg_ua[experiment].shape gives (lat, lon, time), where time=12
        lat, lon, time = uas[model][experiment].shape 
        # Initialize arrays to store seasonal means
        ua_seasons[model][experiment] = np.zeros((lat, lon, 4))
        va_seasons[model][experiment] = np.zeros((lat, lon, 4))
        # Assigning the mean values to each season (0: spring, 1: summer, 2: autumn, 3: winter)
        ua_seasons[model][experiment][:, :, 0] = np.mean(uas[model][experiment][:, :, spr_idx], axis=2)
        ua_seasons[model][experiment][:, :, 1] = np.mean(uas[model][experiment][:, :, sum_idx], axis=2)
        ua_seasons[model][experiment][:, :, 2] = np.mean(uas[model][experiment][:, :, aut_idx], axis=2)
        ua_seasons[model][experiment][:, :, 3] = np.mean(uas[model][experiment][:, :, win_idx], axis=2)
        # same for va    
        va_seasons[model][experiment][:, :, 0] = np.mean(vas[model][experiment][:, :, spr_idx], axis=2)
        va_seasons[model][experiment][:, :, 1] = np.mean(vas[model][experiment][:, :, sum_idx], axis=2)
        va_seasons[model][experiment][:, :, 2] = np.mean(vas[model][experiment][:, :, aut_idx], axis=2)
        va_seasons[model][experiment][:, :, 3] = np.mean(vas[model][experiment][:, :, win_idx], axis=2)

ua_season_diffs = {model: np.zeros((180, 360, 4)) for model in Ensemble_uva}
va_season_diffs = {model: np.zeros((180, 360, 4)) for model in Ensemble_uva}

for model in Ensemble_uva:
    for t in range(len(Seasons)):
        ua_season_diffs[model][:,:,t] = ua_seasons[model]['Eoi400'][:,:,t] - ua_seasons[model]['E280'][:,:,t]
        va_season_diffs[model][:,:,t] = va_seasons[model]['Eoi400'][:,:,t] - va_seasons[model]['E280'][:,:,t]

#%%% 4.2 U, V statistics

# Calculate average of models for each experiment
Expt_avg_ua = {experiment: np.zeros((180, 360, 12)) for experiment in Experiments}
Expt_avg_va = {experiment: np.zeros((180, 360, 12)) for experiment in Experiments}

# Ensemble mean globally
for experiment in Experiments:
    for t in range(len(Time)):
        # Initialise an array to store total OHC across all models:
            Expt_ttl_u = np.zeros((180, 360))
            Expt_ttl_v = np.zeros((180, 360))
            # Iterate over each model
            for model in Ensemble_uva:
                Expt_ttl_u += uas[model][experiment][:,:,t]
                Expt_ttl_v += vas[model][experiment][:,:,t]
            # Calculate average OHC across all models
            Expt_avg_ua[experiment][:,:,t] = Expt_ttl_u / len(Ensemble_uva)
            Expt_avg_va[experiment][:,:,t] = Expt_ttl_v / len(Ensemble_uva)

# Wind anomalies:
Anom_avg_ua = Expt_avg_ua['Eoi400'] - Expt_avg_ua['E280']
Anom_avg_va = Expt_avg_va['Eoi400'] - Expt_avg_va['E280']

# Combine u, v to form vector (magnitude and angle / direction)
# Initialise dictionaries for both these variables
uv_wind_mags = {experiment: np.zeros((180, 360, 12)) for experiment in Experiments}
uv_wind_angs = {experiment: np.zeros((180, 360, 12)) for experiment in Experiments}

# Magnitude and direction of wind
for experiment in Experiments:
    for t in range(len(Time)):
        # Access uas and vas DataArrays for the current model and experiment
        uas_data = Expt_avg_ua[experiment][:,:,t]
        vas_data = Expt_avg_va[experiment][:,:,t]
        # First calculate the magnitude of the wind speed
        uv_wind_mags[experiment][:,:,t] = np.sqrt(uas_data**2 + vas_data**2)
        # Now calculate wind direction
        # Radians
        wind_ang_rad = np.arctan2(vas_data, uas_data)
        # Degrees
        wind_ang_deg = np.degrees(wind_ang_rad)
        # Wind convention, Northward = 0
        uv_wind_angs[experiment][:,:,t] = (90 - wind_ang_deg) % 360


# Initialise arrays for anomaly magnitude and angle
uv_anom_mags = np.zeros((180,360,12))
uv_anom_angs = np.zeros((180,360,12))
# Magnitude and direction of anomaly
for t in range(len(Time)):
    u_anom = Anom_avg_ua[:,:,t]
    v_anom = Anom_avg_va[:,:,t]
    uv_anom_mags[:,:,t] = np.sqrt(u_anom**2 + v_anom**2)
    ang_rad = np.arctan2(v_anom, u_anom)
    ang_deg = np.degrees(ang_rad)
    uv_anom_angs[:,:,t] = (90 - ang_deg) % 360

# Create dictionaries to hold the seasonal data
ua_season = {experiment: np.zeros((lat, lon, 4)) for experiment in Experiments}
va_season = {experiment: np.zeros((lat, lon, 4)) for experiment in Experiments}

# Seasonal calculation for ua and va
for experiment in Experiments:
    # Initialize the array with shape (lat, lon, seasons)
    # Assuming Expt_avg_ua[experiment].shape gives (lat, lon, time), where time=12
    lat, lon, time = Expt_avg_ua[experiment].shape    
    # Assigning the mean values to each season (0: spring, 1: summer, 2: autumn, 3: winter)
    ua_season[experiment][:, :, 0] = np.mean(Expt_avg_ua[experiment][:, :, spr_idx], axis=2)
    ua_season[experiment][:, :, 1] = np.mean(Expt_avg_ua[experiment][:, :, sum_idx], axis=2)
    ua_season[experiment][:, :, 2] = np.mean(Expt_avg_ua[experiment][:, :, aut_idx], axis=2)
    ua_season[experiment][:, :, 3] = np.mean(Expt_avg_ua[experiment][:, :, win_idx], axis=2)
    # same for va    
    va_season[experiment][:, :, 0] = np.mean(Expt_avg_va[experiment][:, :, spr_idx], axis=2)
    va_season[experiment][:, :, 1] = np.mean(Expt_avg_va[experiment][:, :, sum_idx], axis=2)
    va_season[experiment][:, :, 2] = np.mean(Expt_avg_va[experiment][:, :, aut_idx], axis=2)
    va_season[experiment][:, :, 3] = np.mean(Expt_avg_va[experiment][:, :, win_idx], axis=2)

# Combine seasonal u, v to form vector (magnitude and angle / direction)
# Initialise dictionaries for both these variables
uv_season_mags = {experiment: np.zeros((180, 360, 12)) for experiment in Experiments}
uv_season_angs = {experiment: np.zeros((180, 360, 12)) for experiment in Experiments}

for experiment in Experiments:
    for t in range(len(Seasons)):
        # Access uas and vas DataArrays for the current model and experiment
        uas_data1 = ua_season[experiment][:,:,t]
        vas_data1 = va_season[experiment][:,:,t]
        # First calculate the magnitude of the wind speed
        uv_season_mags[experiment][:,:,t] = np.sqrt(uas_data**2 + vas_data**2)
        # Now calculate wind direction
        # Radians
        wind_ang_rad = np.arctan2(vas_data, uas_data)
        # Degrees
        wind_ang_deg = np.degrees(wind_ang_rad)
        # Wind convention, Northward = 0
        uv_season_angs[experiment][:,:,t] = (90 - wind_ang_deg) % 360

# Seasonal anomalies
ua_anom_seasons = np.zeros((lat, lon, 4))
va_anom_seasons = np.zeros((lat, lon, 4))

ua_anom_seasons[:,:,0] = np.mean(Anom_avg_ua[:, :, spr_idx], axis=2)
ua_anom_seasons[:,:,1] = np.mean(Anom_avg_ua[:, :, sum_idx], axis=2)
ua_anom_seasons[:,:,2] = np.mean(Anom_avg_ua[:, :, aut_idx], axis=2)
ua_anom_seasons[:,:,3] = np.mean(Anom_avg_ua[:, :, win_idx], axis=2)
va_anom_seasons[:,:,0] = np.mean(Anom_avg_va[:, :, spr_idx], axis=2)
va_anom_seasons[:,:,1] = np.mean(Anom_avg_va[:, :, sum_idx], axis=2)
va_anom_seasons[:,:,2] = np.mean(Anom_avg_va[:, :, aut_idx], axis=2)
va_anom_seasons[:,:,3] = np.mean(Anom_avg_va[:, :, win_idx], axis=2)

#%% 5. Figures

# lon, lat mesh grid
lonmesh0, latmesh0 = np.meshgrid(Lon, Lat)

# Define Indian Ocean extent (longitude and latitude bounds)
ind_lon_min, ind_lon_max = 20, 140  # Example values, adjust as needed
ind_lat_min, ind_lat_max = -40, 30  # Example values, adjust as needed

# Define Indo-Pacific extent
ind_pac_lon_min, ind_pac_lon_max = 20, 300
ind_pac_lat_min, ind_pac_lat_max = -20, 30

#%%% 5.1 Figure 3cd

fig, axs = plt.subplots(1, 2, figsize=(20, 5), dpi=300, subplot_kw={'projection': cartopy.crs.Robinson(central_longitude=180)})
cf1 = axs[0].contourf(Lon, Lat, t_depth_season_diff[:,:,1], cmap = 'cmo.balance', levels = np.linspace(-110, 110, 12), extend = 'both', transform=cartopy.crs.PlateCarree())
axs[0].set_extent([ind_lon_min, ind_lon_max, ind_lat_min, ind_lat_max], crs=cartopy.crs.PlateCarree())
gl1 = axs[0].gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
gl1.xlocator = mpl.ticker.MaxNLocator(nbins=5)
gl1.ylocator = mpl.ticker.FixedLocator([-30, 0, 30])
gl1.xlabel_style = {'size': 18}
gl1.ylabel_style = {'size': 18}
gl1.top_labels = False
gl1.right_labels = False
# Add stippling where there is significant agreement and no NaN in the ensemble mean change
disagreement_mask = ~sig_agrees_t_depth_seas[:, :, 1] & ~np.isnan(t_depth_season_diff[:, :, 1])
y, x = np.where(disagreement_mask)
x_stipple = lonmesh0[y,x]
y_stipple = latmesh0[y,x]
axs[0].scatter(x_stipple, y_stipple, color='k', s=5, alpha=0.2, marker='o', transform=cartopy.crs.PlateCarree())
cf2 = axs[1].contourf(Lon, Lat, t_depth_season_diff[:,:,3], cmap = 'cmo.balance', levels = np.linspace(-110, 110, 12), extend = 'both', transform=cartopy.crs.PlateCarree())
axs[1].set_extent([ind_lon_min, ind_lon_max, ind_lat_min, ind_lat_max], crs=cartopy.crs.PlateCarree())
gl2 = axs[1].gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
gl2.xlocator = mpl.ticker.MaxNLocator(nbins=5)
gl2.ylocator = mpl.ticker.FixedLocator([-30, 0, 30])
gl2.xlabel_style = {'size': 18}
gl2.ylabel_style = {'size': 18}
gl2.top_labels = False
gl2.right_labels = False
# Add stippling where there is significant agreement and no NaN in the ensemble mean change
disagreement_mask = ~sig_agrees_t_depth_seas[:, :, 3] & ~np.isnan(t_depth_season_diff[:, :, 3])
y, x = np.where(disagreement_mask)
x_stipple = lonmesh0[y,x]
y_stipple = latmesh0[y,x]
axs[1].scatter(x_stipple, y_stipple, color='k', s=5, alpha=0.2, marker='o', transform=cartopy.crs.PlateCarree())
# Create a new axis for the colorbar, positioned below the panels
cbar1 = fig.colorbar(cf1, ax=axs[:])
cbar1.set_label("Δ depth (m)", fontsize=20)
cbar1.ax.tick_params(labelsize=18)
# Set overall plot title and show
#plt.savefig(f"....png", dpi=300, bbox_inches='tight')
#plt.savefig(f"....pdf", dpi=600, bbox_inches='tight')
plt.show()

#%%% 5.2 Figure 3ef

fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=300, subplot_kw={'projection': cartopy.crs.Robinson(central_longitude=180)})
# Winter pi
qv1 = axs[0].quiver(lonmesh0[::5, ::5], latmesh0[::5, ::5], ua_season['E280'][::5, ::5, 3], va_season['E280'][::5, ::5, 3], scale=140, headlength=2, headwidth=4, headaxislength=3, color='k', transform=cartopy.crs.PlateCarree())
# Add quiver key (standard arrow length) in X=0.9, Y=0.1
qk1 = axs[0].quiverkey(qv1, X=0.85, Y=1.05, U=5, label='20 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 18})
#axs[0].set_title(f"$\mathit{{piControl}}$ u, v wind: {Seasons[row_i]}", fontsize=20)
axs[0].coastlines()
axs[0].set_extent([ind_lon_min, ind_lon_max, ind_lat_min, ind_lat_max], crs=cartopy.crs.PlateCarree())
gl1 = axs[0].gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
gl1.xlocator = mpl.ticker.MaxNLocator(nbins=5)
gl1.ylocator = mpl.ticker.FixedLocator([-30, 0, 30])
gl1.xlabel_style = {'size': 18}
gl1.ylabel_style = {'size': 18}
gl1.top_labels = False
gl1.right_labels = False
# Winter anom
qv2 = axs[1].quiver(lonmesh0[::5, ::5], latmesh0[::5, ::5], ua_anom_seasons[::5, ::5, 3], va_anom_seasons[::5, ::5, 3], scale=50, headlength=2, headwidth=4, headaxislength=3, color='k', transform=cartopy.crs.PlateCarree())
# Add quiver key (standard arrow length) in X=0.9, Y=0.1
qk2 = axs[1].quiverkey(qv2, X=0.85, Y=1.05, U=2, label='2 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 18})
#axs[1].set_title(f"Δ u, v wind: {Seasons[row_i]}", fontsize=20)
axs[1].coastlines()
axs[1].set_extent([ind_lon_min, ind_lon_max, ind_lat_min, ind_lat_max], crs=cartopy.crs.PlateCarree())
gl2 = axs[1].gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
gl2.xlocator = mpl.ticker.MaxNLocator(nbins=5)
gl2.ylocator = mpl.ticker.FixedLocator([-30, 0, 30])
gl2.xlabel_style = {'size': 18}
gl2.ylabel_style = {'size': 18}
gl2.top_labels = False
gl2.right_labels = False
# Set overall plot title and show
#plt.savefig(f"....png", dpi=600)#, bbox_inches='tight', bbox_extra_artists=[qk1, qk2])
#plt.savefig(f"....pdf", dpi=600)#, bbox_inches='tight', bbox_extra_artists=[qk1, qk2])
plt.show()

#%%% 5.3 Figure S3

# Create a seasonal figure with 8 subplots (4 rows, 2 columns)
fig, axs = plt.subplots(4, 2, figsize=(18, 24), dpi=300, subplot_kw={'projection': cartopy.crs.Robinson(central_longitude=180)})
# Loop through each season
for row_i in range(4):  # 4 seasons
    # First column: wind anomalies
    ax1 = axs[row_i, 0]
    qv1 = ax1.quiver(lonmesh0[::5, ::5], latmesh0[::5, ::5], ua_season['E280'][::5, ::5, row_i], va_season['E280'][::5, ::5, row_i], scale=200, headlength=2, headwidth=4, headaxislength=3, color='k', transform=cartopy.crs.PlateCarree())
    # Add quiver key (standard arrow length) in X=0.9, Y=0.1
    ax1.quiverkey(qv1,X=0.85, Y= 1.03, U = 20, label = '20 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 18})
    ax1.set_title(f"{Seasons[row_i]}", fontsize=20)
    ax1.coastlines()
    ax1.set_extent([ind_lon_min, ind_lon_max, ind_lat_min, ind_lat_max], crs=cartopy.crs.PlateCarree())
    gl1 = ax1.gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
    gl1.xlocator = mpl.ticker.MaxNLocator(nbins=5)
    gl1.ylocator = mpl.ticker.FixedLocator([-30, 0, 30])
    gl1.xlabel_style = {'size': 18}
    gl1.ylabel_style = {'size': 18}
    gl1.top_labels = False
    gl1.right_labels = False
    # Third column: Difference in precipitation (Experiment[1] - Experiment[0])
    ax2 = axs[row_i, 1]
    qv2 = ax2.quiver(lonmesh0[::5, ::5], latmesh0[::5, ::5], ua_anom_seasons[::5, ::5, row_i], va_anom_seasons[::5, ::5, row_i], scale=50, headlength=2, headwidth=4, headaxislength=3, color='k', transform=cartopy.crs.PlateCarree())
    ax2.quiverkey(qv2,X=0.85, Y= 1.03, U = 2, label = '2 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 18})
    ax2.set_title(f"{Seasons[row_i]}", fontsize=20)
    ax2.coastlines()
    ax2.set_extent([ind_lon_min, ind_lon_max, ind_lat_min, ind_lat_max], crs=cartopy.crs.PlateCarree())
    gl2 = ax2.gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
    gl2.xlocator = mpl.ticker.MaxNLocator(nbins=5)
    gl2.ylocator = mpl.ticker.FixedLocator([-30, 0, 30])
    gl2.xlabel_style = {'size': 18}
    gl2.ylabel_style = {'size': 18}
    gl2.top_labels = False
    gl2.right_labels = False
# Set overall plot title and show
#plt.savefig(f"....png", dpi=300, bbox_inches='tight')
#plt.savefig(f"....pdf", dpi=600, bbox_inches='tight')
plt.show()
