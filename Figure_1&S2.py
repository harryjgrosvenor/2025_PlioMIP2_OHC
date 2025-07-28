# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 09:26:13 2025

@author: Harry J Grosvenor
"""

#%%% 1. Import required packages

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xa
import pandas
import regionmask
import cartopy
import cmocean

#%% 2. Define ensembles and universal variables

# Define the list of experiments
# Experiment nomenclature:
# E280 = piControl
# Eoi400 = midPliocene
Experiments = ["E280", "Eoi400"]

Lat = np.arange(-89.5, 90, 1)
Lon = np.arange(0, 360, 1)

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

# Threshold for stippling
Threshold_frac = 2/3

# Define the list of models:
Ensemble = ["CCSM4", "CCSM4-Utrecht", "CESM1.2", "CESM2", "COSMOS",
                "EC-Earth3-LR", "GISS-E2-1-G", "HadCM3", "HadGEM3", "IPSL-CM6A-LR",
                "MIROC4M", "NorESM-L", "NorESM1-F"]

#%% 3. Data Processing and Statistics

#%%% 3.1 OHC: File read in and define variables

#Control which files to read in
Depth_upp = 0
Depth_low = 700
# Change depth_low to 200 for Figure 1b

# Initialize a dictionary to store datasets for each experiment
datasets_ohcd = {model: {} for model in Ensemble}
datasets_therm = {model: {experiment: {} for experiment in Experiments} for model in Ensemble}
datasets_thermohc = {model: {experiment: {} for experiment in Experiments} for model in Ensemble}

# Loop through each experiment
for model in Ensemble:
    try:
        filepath_ohcd = f"Data/PlioMIP2/Fixed Depth Layers/Diffs_{model}_m2_tarz_OHC_{Depth_upp}_{Depth_low}1x1.nc"
        datasets_ohcd[model] = xa.open_dataset(filepath_ohcd)
    except:
        print(f"OHC Diffs file not found for {model}")
    # Loop through each model within the experiment
    for experiment in Experiments:
        try:
            # Construct file path for the current experiment and model
            filepath_therm = f"Data/PlioMIP2/Thermoclines/{experiment}_{model}_1deg_thermoclines.nc"
            filepath_thermohc = f"Data/PlioMIP2/Thermoclines/{experiment}_{model}_m2_OHC_thermocline1x1.nc"
            # Read in dataset for the current experiment and model
            datasets_therm[model][experiment] = xa.open_dataset(filepath_therm)
            datasets_thermohc[model][experiment] = xa.open_dataset(filepath_thermohc)
        except:
            print(f"OHC file not found for {model} {experiment}")

# Variables for OHC
# Initialise Ocean Heat Content dictionary
ohc_diffs = {model: {} for model in Ensemble}
t_ohcs = {model: {experiment: {} for experiment in Experiments} for model in Ensemble}
t_thetas = {model: {experiment: {} for experiment in Experiments} for model in Ensemble}
t_depths = {model: {experiment: {} for experiment in Experiments} for model in Ensemble}

# Define Ocean Heat Content variable for each model and experiment
for model in Ensemble:
    ohc_diffs[model] = datasets_ohcd[model].OHC
    for experiment in Experiments:
        t_ohcs[model][experiment] = datasets_thermohc[model][experiment].OHC
        t_depths[model][experiment] = datasets_therm[model][experiment].depth

#%%% 3.2 OHC: Statistics

# Calculate difference between experiments for each model
Expt_diff_OHC = {model: np.zeros((180,360)) for model in Ensemble}
Expt_diff_t_ohcs = {model: np.zeros((180,360)) for model in Ensemble}
Expt_diff_t_depths = {model: np.zeros((180,360)) for model in Ensemble}
Expt_diff_t_temps = {model: np.zeros((180,360)) for model in Ensemble}

for model in Ensemble:
    Expt_diff_OHC[model] = ohc_diffs[model]
    Expt_diff_t_ohcs[model] = t_ohcs[model]['Eoi400'] - t_ohcs[model]['E280']
    Expt_diff_t_depths[model] = t_depths[model]['Eoi400'] - t_depths[model]['E280']
    
# Stippling work:
Expt_diff_OHC_array = np.array([Expt_diff_OHC[model].data for model in Ensemble])
Expt_diff_t_ohc_array = np.array([Expt_diff_t_ohcs[model].data for model in Ensemble])
Expt_diff_t_depths_array = np.array([Expt_diff_t_depths[model].data for model in Ensemble])

# shape info
models_ohc, lat, lon = Expt_diff_OHC_array.shape
# Determine direction of change for each model (True for positive, false for negative, NaN for no change)
direc_change_OHC = np.where(np.isnan(Expt_diff_OHC_array), np.nan, Expt_diff_OHC_array > 0)
# Exclude the cases where data stays the same
direc_change_OHC = np.where(Expt_diff_OHC_array == 0, np.nan, direc_change_OHC)
agree_no_plus_OHC = np.nansum(direc_change_OHC ==1, axis = 0)
agree_no_neg_OHC = np.nansum(direc_change_OHC ==0, axis = 0)
# Determine threshold for stippling
threshold_OHC = Threshold_frac * models_ohc
sig_agree_OHC = (agree_no_plus_OHC >= threshold_OHC) | (agree_no_neg_OHC >= threshold_OHC)
        
# Calculate difference between experiments
Expt_avg_OHC_diff = np.nanmean(np.stack([Expt_diff_OHC[model] for model in Ensemble], axis = 0), axis =0)
# nan out consistent land-sea masks
Expt_avg_OHC_diff[pi_msk] = np.nan
Expt_avg_OHC_diff[plio_msk] = np.nan

# Ocean heat content above the thermocline stippling work:
# shape info
models_tohc, lat, lon = Expt_diff_t_ohc_array.shape
# Determine direction of change for each model (True for positive, false for negative, NaN for no change)
direc_change_t_ohc = np.where(np.isnan(Expt_diff_t_ohc_array), np.nan, Expt_diff_t_ohc_array > 0)
# Exclude the cases where data stays the same
direc_change_t_ohc = np.where(Expt_diff_t_ohc_array == 0, np.nan, direc_change_t_ohc)
agree_no_plus_t_ohc = np.nansum(direc_change_t_ohc ==1, axis = 0)
agree_no_neg_t_ohc = np.nansum(direc_change_t_ohc ==0, axis = 0)
# Determine threshold for stippling
threshold_t_ohc = Threshold_frac * models_tohc
sig_agree_t_ohc = (agree_no_plus_t_ohc >= threshold_t_ohc) | (agree_no_neg_t_ohc >= threshold_t_ohc)

# Calculate average of models for each experiment
Expt_avg_t_ohc = {experiment: np.zeros((180, 360)) for experiment in Experiments}
# Ensemble mean globally using nanmean
for experiment in Experiments:
    tohc_ann_stack = np.stack([t_ohcs[model][experiment] for model in Ensemble], axis=0)  # Stack models' data
    Expt_avg_t_ohc[experiment] = np.nanmean(tohc_ann_stack, axis=0)  # Calculate mean ignoring NaNs 
    if experiment == 'Eoi400':
        Expt_avg_t_ohc[experiment][plio_msk] = np.nan
    elif experiment == 'E280':
        Expt_avg_t_ohc[experiment][pi_msk] = np.nan
        
# Calculate difference between experiments
Expt_avg_t_ohc_diff = Expt_avg_t_ohc['Eoi400'] - Expt_avg_t_ohc['E280']

# Thermocline depth stippling work
# shape info
models_t_depth, lat, lon = Expt_diff_t_depths_array.shape
# Determine direction of change for each model (True for positive, false for negative, NaN for no change)
direc_change_t_depth = np.where(np.isnan(Expt_diff_t_depths_array), np.nan, Expt_diff_t_depths_array > 0)
# Exclude the cases where data stays the same
direc_change_t_depth = np.where(Expt_diff_t_depths_array == 0, np.nan, direc_change_t_depth)
agree_no_plus_t_depth = np.nansum(direc_change_t_depth ==1, axis = 0)
agree_no_neg_t_depth = np.nansum(direc_change_t_depth ==0, axis = 0)
# Determine threshold for stippling
threshold_t_depth = Threshold_frac * models_t_depth
sig_agree_t_depth = (agree_no_plus_t_depth >= threshold_t_depth) | (agree_no_neg_t_depth >= threshold_t_depth)
# Calculate average of models for each experiment
Expt_avg_t_depth = {experiment: np.zeros((180, 360)) for experiment in Experiments}


# Ensemble mean globally using nanmean
for experiment in Experiments:
    t_depth_ann_stack = np.stack([t_depths[model][experiment] for model in Ensemble], axis=0)  # Stack models' data
    Expt_avg_t_depth[experiment] = np.nanmean(t_depth_ann_stack, axis=0)  # Calculate mean ignoring NaNs 
    if experiment == 'Eoi400':
        Expt_avg_t_depth[experiment][plio_msk] = np.nan
    elif experiment == 'E280':
        Expt_avg_t_depth[experiment][pi_msk] = np.nan
# Calculate difference between experiments
Expt_avg_t_depth_diff = Expt_avg_t_depth['Eoi400'] - Expt_avg_t_depth['E280']

#%% 4. Figures

#%%% 4.1 Processing for figures

# Indo-Pacific region
lon_ind_pac_min, lon_ind_pac_max = 20, 300
lat_ind_pac_min, lat_ind_pac_max = -40, 30

#%%% 4.2 Figure Plotting

#%%%% 4.2.1 Figure 1

#%%%%% 4.2.1.1 Figure 1a

# OHC: Plot the difference
Fig1a, axs = plt.subplots(1, 1, figsize=(18, 15), dpi=300, subplot_kw={'projection': cartopy.crs.Robinson(central_longitude=180)})
cf1 = axs.contourf(Lon, Lat, Expt_avg_OHC_diff, cmap = 'cmo.balance',levels = np.linspace(-8.5e9, 8.5e9, 18), extend = 'both', transform=cartopy.crs.PlateCarree())
gl1 = axs.gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
gl1.right_labels = False
gl1.top_labels = False
gl1.xlabel_style = {'size': 18}
gl1.ylabel_style = {'size': 18}
gl1.xlocator = mpl.ticker.FixedLocator([60, 180, -60])
gl1.ylocator = mpl.ticker.MaxNLocator(nbins=3)
# Add stippling where there is significant agreement and no NaN in the ensemble mean change
disagreement_mask = ~sig_agree_OHC & ~np.isnan(Expt_avg_OHC_diff)
y, x = np.where(disagreement_mask)
x_stipple = np.interp(x, (0, lon-1), (np.min(Lon), np.max(Lon)))
y_stipple = np.interp(y, (0, lat-1), (np.min(Lat), np.max(Lat)))
axs.scatter(x_stipple, y_stipple, color='k', s=0.5, alpha = 0.5, marker='o', transform=cartopy.crs.PlateCarree())
cbar = plt.colorbar(cf1, ax=axs, shrink=0.5, aspect=20, pad=0.05)
cbar.set_label("Δ OHC (J / m$^2$)", fontsize = 20)
cbar.formatter = mpl.ticker.ScalarFormatter(useMathText=False)
cbar.ax.yaxis.get_offset_text().set(size=18)
cbar.ax.tick_params(labelsize=18)
# Save the figure
#Fig1a.suptitle(f'Ensemble Mean OHC: {Depth_upp} - {Depth_low} m', fontsize = 30, x = 0.435, y = 0.9)
#plt.savefig(f"....png", dpi=600, bbox_inches='tight')
#plt.savefig(f"....pdf", dpi=600, bbox_inches='tight')
plt.show()


#%%%%% 4.2.1.2 Figure 1bcd

Fig1bcd, axs = plt.subplots(3, 1, figsize=(20, 14), dpi=300, subplot_kw={'projection': cartopy.crs.Robinson(central_longitude=180)})
cf1 = axs[0].contourf(Lon, Lat, Expt_avg_OHC_diff, cmap = 'cmo.balance',levels = np.linspace(-1.9e9, 1.9e9, 20), extend = 'both', transform=cartopy.crs.PlateCarree())
axs[0].set_extent([lon_ind_pac_min, lon_ind_pac_max, lat_ind_pac_min, lat_ind_pac_max], crs=cartopy.crs.PlateCarree())
gl1 = axs[0].gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
gl1.right_labels = False
gl1.top_labels = False
gl1.xlabel_style = {'size': 18}
gl1.ylabel_style = {'size': 18}
gl1.xlocator = mpl.ticker.FixedLocator([60, 180, -60])
gl1.ylocator = mpl.ticker.FixedLocator([-30, 0, 30])
# Add stippling where there is significant agreement and no NaN in the ensemble mean change
disagreement_mask = ~sig_agree_OHC & ~np.isnan(Expt_avg_OHC_diff)
y, x = np.where(disagreement_mask)
x_stipple = np.interp(x, (0, lon-1), (np.min(Lon), np.max(Lon)))
y_stipple = np.interp(y, (0, lat-1), (np.min(Lat), np.max(Lat)))
axs[0].scatter(x_stipple, y_stipple, color='k', s=0.5, alpha = 0.5, marker='o', transform=cartopy.crs.PlateCarree())
cbar1 = plt.colorbar(cf1, ax=axs[0], shrink=1, aspect=20, pad=0.05)
cbar1.set_label("Δ OHC (J / m$^2$)", fontsize = 18)
cbar1.formatter = mpl.ticker.ScalarFormatter(useMathText=False)
cbar1.ax.yaxis.get_offset_text().set(size=18)
cbar1.ax.tick_params(labelsize=18)
cf2 = axs[1].contourf(Lon, Lat, Expt_avg_t_ohc_diff, cmap = 'cmo.balance',levels = np.linspace(-5.5e10, 5.5e10, 12), extend = 'both', transform=cartopy.crs.PlateCarree())
#axs.coastlines(linewidth=0.5)
axs[1].set_extent([lon_ind_pac_min, lon_ind_pac_max, lat_ind_pac_min, lat_ind_pac_max], crs=cartopy.crs.PlateCarree())
cbar2 = plt.colorbar(cf2, ax=axs[1], shrink=1, aspect=20, pad=0.05)
cbar2.set_label(r"Δ OHC (J/m$^2$)", fontsize = 18)
cbar2.formatter = mpl.ticker.ScalarFormatter(useMathText=False)
cbar2.ax.yaxis.get_offset_text().set(size=18)
cbar2.ax.tick_params(labelsize=18) 
gl2 = axs[1].gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
gl2.right_labels = False
gl2.top_labels = False
gl2.xlocator = mpl.ticker.FixedLocator([60, 180, -60])
gl2.ylocator = mpl.ticker.FixedLocator([-30, 0, 30])
gl2.xlabel_style = {'size': 18}
gl2.ylabel_style = {'size': 18}
#axs.set_title(f"Δ OHC within {Depth_upp} - {Depth_low} m depth layer", fontsize = 22)
# Add stippling where there is significant agreement and no NaN in the ensemble mean change
disagreement_mask = ~sig_agree_t_ohc & ~np.isnan(Expt_avg_t_ohc_diff)
y, x = np.where(disagreement_mask)
x_stipple = np.interp(x, (0, lon-1), (np.min(Lon), np.max(Lon)))
y_stipple = np.interp(y, (0, lat-1), (np.min(Lat), np.max(Lat)))
axs[1].scatter(x_stipple, y_stipple, color='k', s=2, alpha = 0.5, marker='o', transform=cartopy.crs.PlateCarree())
cf3 = axs[2].contourf(Lon, Lat, Expt_avg_t_depth_diff, cmap = 'cmo.balance',levels = np.linspace(-55, 55, 12), extend = 'both', transform=cartopy.crs.PlateCarree())
#axs.coastlines(linewidth=0.5)
axs[2].set_extent([lon_ind_pac_min, lon_ind_pac_max, lat_ind_pac_min, lat_ind_pac_max], crs=cartopy.crs.PlateCarree())
cbar3 = plt.colorbar(cf3, ax=axs[2], shrink=1, aspect=20, pad=0.05)
cbar3.set_label("Δ depth (m)", fontsize = 18)
cbar3.formatter = mpl.ticker.ScalarFormatter(useMathText=False)
cbar3.ax.yaxis.get_offset_text().set(size=18)
cbar3.ax.tick_params(labelsize=18) 
gl3 = axs[2].gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
gl3.right_labels = False
gl3.top_labels = False
gl3.xlabel_style = {'size': 18}
gl3.ylabel_style = {'size': 18}
gl3.xlocator = mpl.ticker.FixedLocator([60, 180, -60])
gl3.ylocator = mpl.ticker.FixedLocator([-30, 0, 30])
#axs.set_title(f"Δ OHC within {Depth_upp} - {Depth_low} m depth layer", fontsize = 22)
# Add stippling where there is significant agreement and no NaN in the ensemble mean change
disagreement_mask = ~sig_agree_t_depth & ~np.isnan(Expt_avg_t_depth_diff)
y, x = np.where(disagreement_mask)
x_stipple = np.interp(x, (0, lon-1), (np.min(Lon), np.max(Lon)))
y_stipple = np.interp(y, (0, lat-1), (np.min(Lat), np.max(Lat)))
axs[2].scatter(x_stipple, y_stipple, color='k', s=2, alpha = 0.5, marker='o', transform=cartopy.crs.PlateCarree())
# Save the figure
#Fig1bcd.suptitle(f'Ensemble Mean OHC: {Depth_upp} - {Depth_low} m', fontsize = 30, x = 0.435, y = 0.9)
#plt.savefig(f"....png", dpi=300, bbox_inches='tight')
#plt.savefig(f"....pdf", dpi=600, bbox_inches='tight')
plt.show()

#%%%% 4.2.2 Figure S1

# Difference between Eoi400 and control
FigS2, axs = plt.subplots(7, 2, figsize=(20, 32), dpi=300, subplot_kw={'projection': cartopy.crs.Robinson(central_longitude = 180)})
for i, model in enumerate(Ensemble):
    row_i = i // 2
    col_i = i % 2
    cf = axs[row_i, col_i].contourf(Lon, Lat, ohc_diffs[model], cmap = 'cmo.balance',levels = np.linspace(-1e10, 1e10, 21), extend = 'both', transform=cartopy.crs.PlateCarree())
    axs[row_i, col_i].set_title(f"{model}", fontsize = 20)
    gl = axs[row_i, col_i].gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}
    gl.xlocator = mpl.ticker.FixedLocator([60, 180, -60])
    gl.ylocator = mpl.ticker.MaxNLocator(nbins=3)
    #axs[row_i, col_i].coastlines()
cbar = FigS2.colorbar(cf, ax=axs, shrink=0.6, aspect=20, pad=0.05)
cbar.set_label("Difference in OHC (J / m$^2$)", fontsize = 20)
cbar.ax.tick_params(labelsize=20) 
axs[6,1].remove()
#FigS2.suptitle('OHC in (Eoi400 - E280)', fontsize = 24, x = 0.435, y = 0.9)
#plt.savefig(f"....png", dpi=300, bbox_inches='tight')
#plt.savefig(f"....pdf", dpi=600, bbox_inches='tight')
plt.show()
