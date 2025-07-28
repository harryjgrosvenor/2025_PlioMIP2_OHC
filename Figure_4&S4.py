# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 09:08:41 2025

@author: Harry J Grosvenor
"""

#%% 1. Import packages

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xa
import pandas as pd
import pandas
import regionmask
import cartopy
import cmocean

#%%% 1.1 Define functions

# For rounding OHC totals that accompany figures
def round_sig(x, sig=3):
    from math import log10, floor
    if x == 0:
        return 0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

#%% 2. Read in files and create dictionaries for ensemble and experiments

# Define files to be read in based on depths to be analysed
depth_upp1 = 0
depth_low1 = 700

Lat = np.arange(-89.5, 90, 1)
Lon = np.arange(0, 360, 1)

# Define the list of CMIP6 models:
Ensemble = ["CESM2", "EC-Earth3-LR", "GISS-E2-1-G", "HadGEM3", "IPSL-CM6A-LR"]

# Define the list of CMIP6 experiments
Experiments_fut = {
    'historical': {
        'type': 'historical',
        'periods': ['1850_1900']
    },
    'ssp126': {
        'type': 'future',
        'science': 'ssp1-2.6',
        'periods': ['2021_2040', '2041_2060', '2081_2100']
    },
    'ssp245': {
        'type': 'future',
        'science': 'ssp2-4.5',
        'periods': ['2021_2040', '2041_2060', '2081_2100']
    },
    'ssp585': {
        'type': 'future',
        'science': 'ssp5-8.5',
        'periods': ['2021_2040', '2041_2060', '2081_2100']
    }
}


# Initialize a dictionary to store datasets
datasets_fut = {experiment: {} for experiment in Experiments_fut}
datasets_plio = {model: {} for model in Ensemble}


# Loop through each model
for experiment, details in Experiments_fut.items():
    for period in details['periods']:
        # Skip if the model is missing for this experiment
        datasets_fut[experiment][period] = {}
        for model in Ensemble:
            file_path1 = f"Data/ScenarioMIP/{experiment}_{period}_{model}_m2_tarz_OHC_{depth_upp1}_{depth_low1}1x1.nc"
            try:
                # Attempt to open dataset
                datasets_fut[experiment][period][model] = xa.open_dataset(file_path1)
            except FileNotFoundError:
                print(f"Warning: File not found - {file_path1}")
                datasets_fut[experiment][period][model] = None  # Store as None if missing
            except Exception as e:
                print(f"Error loading {file_path1}: {e}")
                datasets_fut[experiment][period][model] = None               

for model in Ensemble:
    file_path1 = f"Data/PlioMIP2/Fixed Depth Layers/diffs_{model}_m2_tarz_OHC_{depth_upp1}_{depth_low1}1x1.nc"
    datasets_plio[model] = xa.open_dataset(file_path1)

# Land-sea masks for experiments
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

# Variables for OHC
# Initialise Ocean Heat Content dictionary for futures
ohcs_fut = {experiment: {period: {} for period in details['periods']} for experiment, details in Experiments_fut.items()}
# For pliocene
ohcs_plio = {model: {} for model in Ensemble}

# Define Ocean Heat Content variable for each model and experiment
for experiment, details in Experiments_fut.items():
    for period in details['periods']:
        for model in Ensemble:
            ohcs_fut[experiment][period][model] = datasets_fut[experiment][period][model].OHC         

# no experiments for pliocene since reading in difference only.
for model in Ensemble:
    ohcs_plio[model] = datasets_plio[model].OHC

# Read in xlsx data with ohc calculations from native grids:
file_path4 = f"Data/OHC_fut_totals_output_{depth_upp1}_{depth_low1}.xlsx"    
file_path5 = f"Data/OHC_diff_totals_output_glob.xlsx"

calcs_fut = pd.read_excel(file_path4, f"OHC {depth_upp1}-{depth_low1} m depth")
calcs_plio = pd.read_excel(file_path5, f"OHC diffs {depth_upp1}-{depth_low1} m depth")

#%% 3. Processing and Statistics

#%%% 3.1 Universal statistics and constnat values

# Thresholds agreement
Threshold_frac0 = 2/3

#%%% 3.2 OHC processing and statistics

ohc_diff_fut = {experiment: {period: {} for period in details['periods']}
    for experiment, details in Experiments_fut.items() if details['type'] == 'future'}

# Loop over future experiments and their periods
for experiment, details in Experiments_fut.items():
    if details['type'] == 'future':  # Only process future experiments
        for period in details['periods']:
            for model in Ensemble:  # Loop over models
                # Load historical data (same for all models)
                historical_data = ohcs_fut['historical']['1850_1900'][model]
                # Load future data for the current experiment, period, and model
                future_data = ohcs_fut[experiment][period][model]
                # Compute the difference (Future - Historical)
                ohc_diff_fut[experiment][period][model] = future_data - historical_data

# Stippling work:
# Convert ohc_diff_fut into a dictionary of NumPy arrays
ohc_diff_fut_array = {
    experiment: {
        period: np.array([ohc_diff_fut[experiment][period][model] for model in Ensemble])
        for period in Experiments_fut[experiment]['periods']}
    for experiment in ohc_diff_fut}

stippling_fut = {}

# shape info
for experiment in ohc_diff_fut:
    stippling_fut[experiment] = {}
    for period in Experiments_fut[experiment]['periods']:
        models_ohc, lat, lon = ohc_diff_fut_array[experiment][period].shape
        # Determine direction of change for each model (True for positive, false for negative, NaN for no change)
        direc_change_ohc = np.where(np.isnan(ohc_diff_fut_array[experiment][period]), np.nan, ohc_diff_fut_array[experiment][period] > 0)
        # Exclude the cases where data stays the same
        direc_change_ohc = np.where(ohc_diff_fut_array[experiment][period] == 0, np.nan, direc_change_ohc)
        agree_no_plus_ohc = np.nansum(direc_change_ohc ==1, axis = 0)
        agree_no_neg_ohc = np.nansum(direc_change_ohc ==0, axis = 0)
        # Determine threshold for stippling
        threshold_ohc = Threshold_frac0 * models_ohc
        sig_agree_ohc = (agree_no_plus_ohc >= threshold_ohc) | (agree_no_neg_ohc >= threshold_ohc)
        stippling_fut[experiment][period] = sig_agree_ohc

# Calculate average of models for each experiment, period
ohc_fut_avg = {experiment: {period: np.zeros((180, 360)) for period in details['periods']}
    for experiment, details in Experiments_fut.items()}

# Ensemble mean globally using nanmean
for experiment in Experiments_fut:
    for period in Experiments_fut[experiment]['periods']:
        ohc_ann_stack = np.stack([ohcs_fut[experiment][period][model] for model in Ensemble], axis=0)  # Stack models' data
        ohc_fut_avg[experiment][period] = np.nanmean(ohc_ann_stack, axis=0)  # Calculate mean ignoring NaNs 

# Ensemble mean in pliocene
ohc_ann_stack = np.stack([ohcs_plio[model] for model in Ensemble], axis = 0)
ohc_plio_avg = np.nanmean(ohc_ann_stack, axis = 0)
ohc_plio_avg = np.where(ohc_plio_avg==0, np.nan, ohc_plio_avg)
ohc_plio_avg[plio_msk] = np.nan
ohc_plio_avg[pi_msk] = np.nan

# Initialise ohc avg difference dictionary
ohc_fut_avg_diff = {experiment: {period: np.zeros((180, 360)) for period in details['periods']} for experiment, details in Experiments_fut.items() if details['type'] == 'future'}

# Calculate difference between experiments
for experiment, details in Experiments_fut.items():
    if details['type'] == 'future':
        for period in details['periods']:
            ohc_fut_avg_diff[experiment][period] = ohc_fut_avg[experiment][period] - ohc_fut_avg['historical']['1850_1900']

# Define difference between future and Pliocene
ohcs_plio_fut_diffs = {}
ohcs_fut_plio_diffs = {}

for experiment, details in Experiments_fut.items():
    if details['type'] == 'future':
        ohcs_plio_fut_diffs[experiment] = {}
        ohcs_fut_plio_diffs[experiment] = {}
        for period in details['periods']:
            ohcs_plio_fut_diffs[experiment][period] = {}
            ohcs_fut_plio_diffs[experiment][period] = {}
            for model in Ensemble:
                ohcs_plio_fut_diffs[experiment][period][model] = ohcs_plio[model] - ohc_diff_fut[experiment][period][model]
                ohcs_fut_plio_diffs[experiment][period][model] = ohc_diff_fut[experiment][period][model] - ohcs_plio[model]

# Stippling for Pliocene - future differences            
# Convert ohc_fut_plio into a dictionary of NumPy arrays
ohc_plio_fut_array = {
    experiment: {
        period: np.array([ohcs_plio_fut_diffs[experiment][period][model] for model in Ensemble])
        for period in Experiments_fut[experiment]['periods']}
    for experiment in ohc_diff_fut}

stippling_pliofut = {}

# shape info
for experiment in ohc_diff_fut:
    stippling_pliofut[experiment] = {}
    for period in Experiments_fut[experiment]['periods']:
        models_ohc, lat, lon = ohc_plio_fut_array[experiment][period].shape
        # Determine direction of change for each model (True for positive, false for negative, NaN for no change)
        direc_change_pliofut = np.where(np.isnan(ohc_plio_fut_array[experiment][period]), np.nan, ohc_plio_fut_array[experiment][period] > 0)
        # Exclude the cases where data stays the same
        direc_change_pliofut = np.where(ohc_plio_fut_array[experiment][period] == 0, np.nan, direc_change_pliofut)
        agree_no_plus_pliofut = np.nansum(direc_change_pliofut ==1, axis = 0)
        agree_no_neg_pliofut = np.nansum(direc_change_pliofut ==0, axis = 0)
        # Determine threshold for stippling
        threshold_pliofut = Threshold_frac0 * models_ohc
        sig_agree_pliofut = (agree_no_plus_pliofut >= threshold_pliofut) | (agree_no_neg_pliofut >= threshold_pliofut)
        stippling_pliofut[experiment][period] = sig_agree_pliofut

#%%% 3.3 Excel sheet processing

# Futures sheet
fut_models_xl = {experiment: {period: {model: None for model in Ensemble} for period in details['periods']}
    for experiment, details in Experiments_fut.items() if details['type'] == 'future'}

fut_mn_xl = {experiment: {period: None for period in details['periods']}
    for experiment, details in Experiments_fut.items() if details['type'] == 'future'}

for experiment, details in Experiments_fut.items():
    if details['type'] == 'future':  # Only process future experiments
        for period in details['periods']:
            for model in Ensemble:
                value1 = calcs_fut.loc[calcs_fut['Model'] == model, f"{experiment} {period} tarz OHC (J)"]
                value2 = calcs_fut.loc[calcs_fut['Model'] == model, "historical 1851_1900 tarz OHC (J)"]
                fut_models_xl[experiment][period][model] = value1.values - value2.values
            fut_mn_xl[experiment][period] = np.nanmean(list(fut_models_xl[experiment][period].values()))

# Pliocene sheet
# Initialise store for pliocene calculations
plio_models_xl = {model: None for model in Ensemble}

# Pick out the models from this analysis
for model in Ensemble:
    value = calcs_plio.loc[calcs_plio['Model'] == model, 'OHC Diff in pi Ocean (J)']    
    # If a value exists, store it in the dictionary
    if not value.empty:
        plio_models_xl[model] = value.values[0]  # Extract the scalar value
    else:
        plio_models_xl[model] = np.nan  # Store NaN if the model is missing (optional)

# Calculate mean
plio_mn_xl = np.nanmean(list(plio_models_xl.values()))

#%% 4. Figures

#%%% 4.1 Figure 4

# Create a 3-row, 2-column subplot layout
fig, axs = plt.subplots(3, 2, figsize=(12, 15), dpi=300, subplot_kw={'projection': cartopy.crs.Robinson(central_longitude=180)})
axs = axs.flatten()  # Flatten for easier indexing
# Iterate over the experiments (columns in original, rows now)
plot_index = 0  # Keep manual control over subplot index
for experiment, details in Experiments_fut.items():
    if details['type'] == 'future':
        science = details['science']
        # Row 1: Future OHC Difference
        cf1 = axs[plot_index].contourf(
            Lon, Lat, ohc_fut_avg_diff[experiment]['2081_2100'],
            cmap='cmo.balance', levels=np.linspace(-6e9, 6e9, 13),
            extend='both', transform=cartopy.crs.PlateCarree())
        axs[plot_index].set_title(f"{science}", fontsize=14)
        # Stippling
        sig_agree = stippling_fut[experiment]['2081_2100']
        disagreement_mask = ~sig_agree & ~np.isnan(ohc_fut_avg_diff[experiment]['2081_2100'])
        y, x = np.where(disagreement_mask)
        x_stipple = np.interp(x, (0, lon - 1), (np.min(Lon), np.max(Lon)))
        y_stipple = np.interp(y, (0, lat - 1), (np.min(Lat), np.max(Lat)))
        axs[plot_index].scatter(x_stipple, y_stipple, color='k', s=0.01, alpha=0.4,
                                marker='o', transform=cartopy.crs.PlateCarree())
        axs[plot_index].text(1.0, -0.22, f"Δ OHC: {round_sig((fut_mn_xl[experiment][period])*(1e-21), 3):.0f} ZJ",
                             transform=axs[plot_index].transAxes, ha='right', va='bottom', fontsize=12, color='black')
        gl1 = axs[plot_index].gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
        gl1.top_labels = False
        gl1.right_labels = False
        gl1.xlocator = mpl.ticker.FixedLocator([-60, 180, 60])
        gl1.ylocator = mpl.ticker.MaxNLocator(nbins=3)
        gl1.xlabel_style = {'size': 10}
        gl1.ylabel_style = {'size': 10}
        plot_index += 1
        # Row 2: Plio - Future OHC Difference
        cf2 = axs[plot_index].contourf(
            Lon, Lat, ohc_fut_avg_diff[experiment]['2081_2100'] - ohc_plio_avg,
            cmap='cmo.balance', levels=np.linspace(-6e9, 6e9, 13),
            extend='both', transform=cartopy.crs.PlateCarree())
        axs[plot_index].set_title(f"{science} - PlioMIP2", fontsize=14)
        sig_agree = stippling_pliofut[experiment]['2081_2100']
        disagreement_mask = ~sig_agree & ~np.isnan(ohc_fut_avg_diff[experiment]['2081_2100'] - ohc_plio_avg)
        y, x = np.where(disagreement_mask)
        x_stipple = np.interp(x, (0, lon - 1), (np.min(Lon), np.max(Lon)))
        y_stipple = np.interp(y, (0, lat - 1), (np.min(Lat), np.max(Lat)))
        axs[plot_index].scatter(x_stipple, y_stipple, color='k', s=0.01, alpha=0.4,
                                marker='o', transform=cartopy.crs.PlateCarree())
        axs[plot_index].text(1.0, -0.22, f"Δ OHC: {round_sig((fut_mn_xl[experiment][period] - plio_mn_xl)*(1e-21),3):.0f} ZJ",
                             transform=axs[plot_index].transAxes, ha='right', va='bottom', fontsize=12, color='black')
        gl2 = axs[plot_index].gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
        gl2.top_labels = False
        gl2.right_labels = False
        gl2.xlocator = mpl.ticker.FixedLocator([-60, 180, 60])
        gl2.ylocator = mpl.ticker.MaxNLocator(nbins=3)
        gl2.xlabel_style = {'size': 10}
        gl2.ylabel_style = {'size': 10}
        plot_index += 1
# Add a colorbar
cbar = fig.colorbar(cf1, ax=axs, orientation='horizontal', shrink=0.8, aspect=40, pad=0.08)
cbar.set_label("OHC (J / m$^2$)", fontsize=12)
cbar.ax.tick_params(labelsize=12)
# Show or save figure
#plt.savefig(f"....png", dpi = 600, bbox_inches='tight')
#plt.savefig(f"....pdf", dpi = 600, bbox_inches='tight')
plt.show()

#%%% 4.2 Figure S4

# Will need to change depth_upp and depth_low at line 33

# Then uncomment

# # Create a 3-row, 1-column subplot layout
# fig, axs = plt.subplots(3, 1, figsize=(12, 15), dpi=300, subplot_kw={'projection': cartopy.crs.Robinson(central_longitude=180)})
# axs = axs.flatten()  # Flatten for easier indexing
# # Iterate over the experiments (columns in original, rows now)
# plot_index = 0  # Keep manual control over subplot index
# for experiment, details in Experiments_fut.items():
#     if details['type'] == 'future':
#         science = details['science']
#         # Row 1: Future OHC Difference
#         cf1 = axs[plot_index].contourf(
#             Lon, Lat, ohc_fut_avg_diff[experiment]['2081_2100'],
#             cmap='cmo.balance', levels=np.linspace(-3e9, 3e9, 13),
#             extend='both', transform=cartopy.crs.PlateCarree())
#         axs[plot_index].set_title(f"{science}", fontsize=14)
#         # Stippling
#         sig_agree = stippling_fut[experiment]['2081_2100']
#         disagreement_mask = ~sig_agree & ~np.isnan(ohc_fut_avg_diff[experiment]['2081_2100'])
#         y, x = np.where(disagreement_mask)
#         x_stipple = np.interp(x, (0, lon - 1), (np.min(Lon), np.max(Lon)))
#         y_stipple = np.interp(y, (0, lat - 1), (np.min(Lat), np.max(Lat)))
#         axs[plot_index].scatter(x_stipple, y_stipple, color='k', s=0.01, alpha=0.4,
#                                 marker='o', transform=cartopy.crs.PlateCarree())
#         #axs[plot_index].text(1.0, -0.22, f"Δ OHC: {round_sig((fut_mn_xl[experiment][period])*(1e-21), 3):.0f} ZJ",
#         #                     transform=axs[plot_index].transAxes, ha='right', va='bottom', fontsize=12, color='black')
#         gl1 = axs[plot_index].gridlines(linewidth=0.5, draw_labels=True, crs=cartopy.crs.PlateCarree())
#         gl1.top_labels = False
#         gl1.right_labels = False
#         gl1.xlocator = mpl.ticker.FixedLocator([-60, 180, 60])
#         gl1.ylocator = mpl.ticker.MaxNLocator(nbins=3)
#         gl1.xlabel_style = {'size': 10}
#         gl1.ylabel_style = {'size': 10}
#         plot_index += 1
# # Add a colorbar
# cbar = fig.colorbar(cf1, ax=axs, orientation='horizontal', shrink=0.8, aspect=40, pad=0.08)
# cbar.set_label("OHC (J / m$^2$)", fontsize=12)
# cbar.ax.tick_params(labelsize=12)
# # Show or save figure
# #plt.savefig(f"....png", dpi = 300, bbox_inches='tight')
# #plt.savefig(f"....pdf", dpi = 600, bbox_inches='tight')
# plt.show()