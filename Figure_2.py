# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:20:15 2025

@author: Harry J Grosvenor
"""

#%%% 1. Import required packages

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as mticker
from matplotlib.transforms import Transform
from matplotlib.scale import ScaleBase, register_scale
import numpy as np
import xarray as xa
import pandas
import regionmask
import scipy
import cartopy
import cmocean

#%% 2. Read in Files

# Define model ensemble (files to be read)
Ensemble = ["CCSM4", "CCSM4-Utrecht", "CESM1.2", "CESM2", "COSMOS", "EC-Earth3-LR",
        "GISS-E2-1-G", "HadCM3", "HadGEM3", "IPSL-CM6A-LR", "MIROC4M",
        "NorESM-L", "NorESM1-F"]

# Define the list of experiments
# Experiment nomenclature:
# E280 = piControl
# Eoi400 = midPliocene
Experiments = ["E280", "Eoi400"]

# Initialize a dictionary to store datasets for each model and experiment
datasets = {model: {} for model in Ensemble}
datasets_therm = {model: {} for model in Ensemble}

# Loop through each experiment
for model in Ensemble:
    # Loop through each model within the experiment
    for experiment in Experiments:
        # Construct file path for the current experiment and model
        filepath = f"Data/{experiment}_{model}_thetao_1x1.nc"
        # Read in dataset for the current experiment and model
        datasets[model][experiment] = xa.open_dataset(filepath)

for model in Ensemble:
    for experiment in Experiments:
        filepath1 = f"Data/PlioMIP2/Thermoclines/{experiment}_{model}_1deg_thermoclines.nc"
        datasets_therm[model][experiment] = xa.open_dataset(filepath1)
        
#%% 3. Variables

# Initialise empty Thetas dictionary to store potential temperature data
Thetas = {model: {} for model in Ensemble}
dt_depth = {model: {} for model in Ensemble}
lat = np.arange(-89.5, 90, 1)
lon = np.arange(0, 360, 1)
Depths = {model: {} for model in Ensemble}

# Assign the correct variable name using our pre-defined function
for model in Ensemble:
    for experiment in Experiments:
        # Define thetao variable for model experiments
        Thetas[model][experiment] = datasets[model][experiment].thetao
        dt_depth[model][experiment] = datasets_therm[model][experiment].depth
        # Eliminate unnecessary dimension of size 1 in Theta
        Thetas[model][experiment] = np.squeeze(Thetas[model][experiment])
        
# Lat, Lon and Depth will all have the same grid structure between experiments so only need one array:        
for model in Ensemble:
    Depths[model] = datasets[model]['Eoi400'].depth # accidentally stripped units off two of the models E280 depths. Eoi400 fine.

# Some models have the depth variable measured in centimeters, convert to meters.
for model in Ensemble:
    if Depths[model].units =='centimeters' or Depths[model].units == 'cm':
        Depths_m = Depths[model] / 100
        if 'z_bounds' in datasets[model]['E280'].variables:
            z_bounds_m = datasets[model]['E280']['z_bounds'].values / 100
            datasets[model]['E280']['z_bounds'].values = z_bounds_m
    else:
        Depths_m = Depths[model]
    Depths[model] = Depths_m

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
pi_lsm = regionmask.defined_regions.natural_earth_v5_1_2.land_110.mask(lon, lat)
pi_msk = ~pi_lsm.isnull()

#%% 4. Constants and target choices

# Choices
laty = 0 # +- span either side
span = 5

# Constants
R_earth = 6.371e6 # Earth's radius in meters

#%% 5. Pre-interpolation Processing

# Initialise dictionary for selected Potential Temps
Thetas_sel = {model: {experiment: {} for experiment in Experiments} for model in Ensemble}
Thetas_zon = {model: {experiment: np.full((len(Depths[model]), len(lon)), np.nan) for experiment in Experiments} for model in Ensemble}
dt_depth_sel = {model: {experiment: {} for experiment in Experiments} for model in Ensemble}
dt_depth_zon = {model: {experiment: np.full((len(lon)), np.nan) for experiment in Experiments} for model in Ensemble}

# Define latitude limits based upon choices
lat_lims = [laty-span, laty+span]

# Select data within target lats
for model in Ensemble:
    for experiment in Experiments:
        # Thetas has shape (depth, lat, lon), so we'll apply the mask to the latitude dimension
        Thetas_sel[model][experiment] = Thetas[model][experiment].copy()  # Create a copy to avoid modifying original
        Thetas_sel[model][experiment][:, ~((lat >= lat_lims[0]) & (lat <= lat_lims[1])), :] = np.nan  # Set latitudes outside the mask to np.nan
        dt_depth_sel[model][experiment] = dt_depth[model][experiment].copy()
        dt_depth_sel[model][experiment][~((lat >= lat_lims[0]) & (lat <= lat_lims[1])), :] = np.nan


# Account for curvature of the earth in the mean
# Define latitude weights (cosine of latitude) for weighted mean
lat_weights = np.where((lat >= lat_lims[0]) & (lat <= lat_lims[1]), np.cos(np.radians(lat)), np.nan)

# Zonal mean across selected data
for model in Ensemble:
    for experiment in Experiments:
        lat_wgts_th = lat_weights[None, :, None]
        lat_wgts_dt = lat_weights[:, None]
        weighted_data = Thetas_sel[model][experiment] * lat_wgts_th # Broadcast lat_weights across depth and longitude
        weighted_therm = dt_depth_sel[model][experiment] * lat_wgts_dt
        lat_mask_th = ~np.isnan(Thetas_sel[model][experiment])
        lat_mask_dt = ~np.isnan(dt_depth_sel[model][experiment])
        lat_wgts_th = np.where(lat_mask_th, lat_wgts_th, np.nan)
        lat_wgts_dt = np.where(lat_mask_dt, lat_wgts_dt, np.nan)
        for i in range(len(lon)):
            for z in range(len(Depths[model])):
                Thetas_zon[model][experiment][z, i] = np.nansum(weighted_data[z,:, i]) / np.nansum(lat_wgts_th[z,:, i])  # Normalize by the sum of the weights
            dt_depth_zon[model][experiment][i] = np.nansum(weighted_therm[:,i]) / np.nansum(np.squeeze(lat_wgts_dt[:,i]))


#%% 6. Interpolating onto standard lon-depth grid

# Need to define a standard set of depths
std_depths = np.concatenate([np.arange(0, 200, 5),
                             np.arange(200, 400, 10),
                             np.arange(400, 750, 25),
                             np.arange(750, 1000, 50),
                             np.arange(1000, 2001, 100)])

Thetas_intp = {model: {} for model in Ensemble}

# Now need to interpolate temperature onto this standard set of depths
for model in Ensemble:
    for experiment in Experiments:
        depths = Depths[model].values
        temps = Thetas_zon[model][experiment]
        temps_intp = np.full((len(std_depths), len(lon)), np.nan)
        for i in range(len(lon)):
            temp_prof = temps[:, i]
            valid_mask = ~np.isnan(temp_prof)
            if np.any(valid_mask):
                # Ensure the first native depth is a scalar (NumPy)
                first_native_depth = depths[0]             
                # Handle the case where some std_depths are shallower than the first native depth
                for j, std_depth in enumerate(std_depths):
                    if std_depth < first_native_depth:
                        # For std_depths shallower than the first native depth, use nearest neighbor
                        temps_intp[j, i] = temp_prof[0]  # Assign the temperature at the first available depth
                    else:
                        # Otherwise, use linear interpolation for remaining depths
                        interp = scipy.interpolate.interp1d(depths, temp_prof, kind='linear', bounds_error=False, fill_value=np.nan)
                        temps_intp[j, i] = interp(std_depth)
        # Store the interpolated temperatures
        Thetas_intp[model][experiment] = temps_intp
        
# Set zero values to np.nan
for model in Ensemble:
    for experiment in Experiments:
        Thetas_intp[model][experiment] = np.where(Thetas_intp[model][experiment] == 0, np.nan, Thetas_intp[model][experiment])

#%%% 6.1 LSM conversions

# Need to convert land-sea masks from lat, lon to depth, lon in the same way as we did thetas
lat_ind = np.where((pi_msk.lat.values >= lat_lims[0]) & (pi_msk.lat.values <= lat_lims[1]))[0]

ocean_present_plio = np.all(plio_msk[lat_ind, :], axis = 0)
ocean_present_pi = np.all(pi_msk.values[lat_ind, :], axis = 0)

plio_z_msk = np.tile(ocean_present_plio, (len(std_depths), 1))
pi_z_msk = np.tile(ocean_present_pi, (len(std_depths), 1))


#%% 8. Ensemble mean and other stats

# Define threshold for agreement
threshold_frac = 2/3

#%%% 8.1 Ensemble mean

# Experiment means
Thetas_trp_mn = {experiment: np.zeros((90, 360)) for experiment in Experiments}

for experiment in Experiments:
    # Stack models' data along a new axis (axis=0) to create a 4D array (models, lat, lon, time)
    theta_stack = np.stack([Thetas_intp[model][experiment] for model in Ensemble], axis=0)
    # Compute the mean across models (axis=0), ignoring NaNs, to retain lat, lon, time dimensions
    Thetas_trp_mn[experiment] = np.nanmean(theta_stack, axis=0)

Thetas_trp_mn['E280'][pi_z_msk] = np.nan
Thetas_trp_mn['Eoi400'][plio_z_msk] = np.nan
    
Theta_trp_mn = Thetas_trp_mn['Eoi400'] - Thetas_trp_mn['E280']

# thermocline mean depth
dt_depth_mn = {experiment: np.zeros((360)) for experiment in Experiments}

for experiment in Experiments:
    # Stack models' data along a new axis (axis=0) to create a 4D array (models, lat, lon, time)
    dt_stack = np.stack([dt_depth_zon[model][experiment] for model in Ensemble], axis=0)
    # Compute the mean across models (axis=0), ignoring NaNs, to retain lat, lon, time dimensions
    dt_depth_mn[experiment] = np.nanmean(dt_stack, axis=0)

dt_depth_mn['E280'][pi_z_msk[0, :]] = np.nan
dt_depth_mn['Eoi400'][plio_z_msk[0, :]] = np.nan

#%%% 8.2 Stippling

Thetas_std_diffs = {model: {} for model in Ensemble}

for model in Ensemble:
    Thetas_std_diffs[model] = Thetas_intp[model]['Eoi400'] - Thetas_intp[model]['E280']  

# convert to array
Thetas_std_diffs_array = np.array([Thetas_std_diffs[model].data for model in Ensemble])

# shape info
models_i, depth_i, lon_i = Thetas_std_diffs_array.shape
# Determine direction of change for each model (True for positive, false for negative, NaN for no change)
direc_change = np.where(np.isnan(Thetas_std_diffs_array), np.nan, Thetas_std_diffs_array > 0)
# Exclude the cases where data stays the same
direc_change = np.where(Thetas_std_diffs_array == 0, np.nan, direc_change)
agree_no_plus = np.nansum(direc_change ==1, axis = 0)
agree_no_neg = np.nansum(direc_change ==0, axis = 0)
# Determine threshold for stippling
threshold = threshold_frac * models_i
sig_agree = (agree_no_plus >= threshold) | (agree_no_neg >= threshold)

#%% 9. Figures

#%%% 9.1 Figure Processing

# Custom scale transformation
def custom_depth_transform(y):
    """Compress y (negative depth) linearly below -200 m"""
    y = np.asarray(y)
    out = np.empty_like(y)
    threshold = -200
    compress_ratio = 0.5  # 1m becomes 0.5m in plotting space
    mask = y >= threshold
    out[mask] = y[mask]
    out[~mask] = threshold + (y[~mask] - threshold) * compress_ratio
    return out

# Inverse for axis ticks
def inverse_custom_depth_transform(y_display):
    threshold = -200
    compress_ratio = 0.5
    y_display = np.asarray(y_display)
    out = np.empty_like(y_display)
    mask = y_display >= threshold
    out[mask] = y_display[mask]
    out[~mask] = threshold + (y_display[~mask] - threshold) / compress_ratio
    return out

class PiecewiseDepthTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    def transform_non_affine(self, y):
        return custom_depth_transform(y)
    def inverted(self):
        return InvertedPiecewiseDepthTransform()

class InvertedPiecewiseDepthTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    def transform_non_affine(self, y):
        return inverse_custom_depth_transform(y)
    def inverted(self):
        return PiecewiseDepthTransform()

class PiecewiseDepthScale(ScaleBase):
    name = 'piecewisedepth'
    def get_transform(self):
        return PiecewiseDepthTransform()
    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(mticker.MultipleLocator(50))
        axis.set_major_formatter(mticker.ScalarFormatter())
    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax
# Register the custom scale
register_scale(PiecewiseDepthScale)

#%%% 9.2 Plotting Figure 2

# Plot with compressed depth-scale
fig, axs = plt.subplots(3, 1, figsize=(16, 24), dpi=300)
# === Panel plotting function ===
def plot_panel(ax, data, title, dt_depth, label_color='silver', contour_levels=[20]):
    pcm = ax.pcolormesh(lon, -std_depths, data, cmap="cmo.thermal", vmin=10, vmax=30, shading='auto')
    ax.set_xlim(40, 280)
    ax.set_ylim(-400, 0)
    ax.set_yscale('piecewisedepth')
    ax.set_xlabel("Longitude East", fontsize=16)
    ax.set_ylabel("Depth (m)", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_title(title, fontsize=18)
    # Contours only — label manually outside this function
    ax.contour(lon, -std_depths, data, levels=[28], colors='k', linewidths=0.8)
    ax.contour(lon, -std_depths, data, levels=contour_levels, colors='k', linewidths=0.8)
    # dt/dz line
    ax.plot(lon, -dt_depth, color=label_color, linewidth=2, label='dt/dz')
    # dt/dz labels
    for label_lon in [130]:
        label_depth = -np.interp(label_lon, lon, dt_depth)
        ax.text(label_lon, label_depth, "dt/dz", fontsize=10, color='k', ha='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    return pcm
# Panel 1: piControl ===
pcm1 = plot_panel(axs[0], Thetas_trp_mn['E280'], "piControl", dt_depth_mn['E280'], label_color='silver', contour_levels=[20])
# Manual contour labels
axs[0].text(90, -45, "28°C", fontsize=10, color='k', ha='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
axs[0].text(125, -150, "20°C", fontsize=10, color='k', ha='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
# Panel 2: midPliocene ===
pcm2 = plot_panel(axs[1], Thetas_trp_mn['Eoi400'], "midPliocene", dt_depth_mn['Eoi400'], label_color='white', contour_levels=[22])
# Manual contour labels
axs[1].text(200, -65, "28°C", fontsize=10, color='k', ha='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
axs[1].text(80, -115, "22°C", fontsize=10, color='k', ha='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
# Shared colorbar for top two panels ===
cbar1 = plt.colorbar(pcm1, ax=axs[0:2])
cbar1.set_label(r"Potential Temperature (°C)", fontsize=16)
cbar1.ax.tick_params(labelsize=16)
# Panel 3: Δ difference ===
cf1 = axs[2].contourf(
    lon, -std_depths, Thetas_trp_mn['Eoi400'] - Thetas_trp_mn['E280'],
    cmap="cmo.balance", levels=np.linspace(-2.6, 2.6, 14), extend='both')
axs[2].set_xlim(40, 280)
axs[2].set_ylim(-400, 0)
axs[2].set_yscale('piecewisedepth')
axs[2].set_xlabel("Longitude East", fontsize=16)
axs[2].set_ylabel("Depth (m)", fontsize=16)
axs[2].tick_params(axis='both', labelsize=16)
axs[2].set_title(r"${\Delta}$", fontsize=18)
# Stippling
disagreement_mask = ~sig_agree & ~np.isnan(Theta_trp_mn)
y, x = np.where(disagreement_mask)
x_stipple = lon[x]
y_stipple = -std_depths[y]
axs[2].scatter(x_stipple, y_stipple, color='k', s=0.5, alpha=0.5, marker='o')
# dt/dz comparison lines
axs[2].plot(lon, -dt_depth_mn['E280'], color='silver', linewidth=2, label='$\mathit{piControl}$')
axs[2].plot(lon, -dt_depth_mn['Eoi400'], color='white', linewidth=2, label='$\mathit{midPliocene}$')
axs[2].legend(loc='center left', bbox_to_anchor=(0.025, 0.1), fontsize=16)
# Colorbar for Δ panel
cbar2 = plt.colorbar(cf1, ax=axs[2])
cbar2.set_label(r"${\Delta}$ Potential Temperature (°C)", fontsize=16)
cbar2.ax.tick_params(labelsize=16)
# Show or save figure
#plt.savefig(f"....png", dpi=600, bbox_inches='tight')
#plt.savefig(f"....pdf", dpi=600, bbox_inches='tight')
plt.show()