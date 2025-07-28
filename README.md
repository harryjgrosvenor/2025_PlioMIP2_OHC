Git repository containing the accompanying code and data for the 2025 Grosvenor et al., paper: "Simulated Ocean Heat Content during the mid-Pliocene Warm Period".

All data analysis was completed using python. Annual and monthly means were computed from each model simulation and where necessary interpolated onto a 1x1 degree regular model grid for multi-model analyses using CDO.

Contents
The code is enough to create the figures from the paper but does not include processing steps.
The data provided is processed data, calculated and processed from the PlioMIP2 and ScenarioMIP climate modelling interomcparison project simulations. This processed data includes netCDFs of ocean heat content calculated over fixed-depth layers; netCDFs of thermocline information; a netCDF for the PRISM4 land-sea mask (Dowsett et al., 2016) and excel spreadsheets for calculation totals.
Data does not include unprocessed data, and particularly, unprocessed thetao and so data which can be accessed from the ESGF and from the PlioMIP2 Globus databases. This data is required for some of the figures.
