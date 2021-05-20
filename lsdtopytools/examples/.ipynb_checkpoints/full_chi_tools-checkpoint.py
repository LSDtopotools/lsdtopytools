"""
This file provide a commented example going through all the range of tools related to chi extraction, ksn, knickpoints and things like that.
It will become a rather massive file and aim to give a large overview of what we can do with this tools. For more specific use (e.g., focusing on caoncavity)
you can refer to the more specific examples (if it exist yet).
Authors: B.G. 
"""

# The following lines import the module we will use. Any python scripts starts with importing other bits of codes we need
from lsdtopytools import LSDDEM # I am telling python I will need this module to run.
from lsdtopytools import quickplot, quickplot_ksn_knickpoints, quickplot_movern # from the same tools, I am importing some quick plotting modules
# You can add any extra dependencies you need here. For example pandas, numpy or matplotlib.

# First step is to load the DEM
## The name "mydem" can be changed into whatever suits you
## Let's have a clean and organised approach and save things into variables
path_to_dem = "/adapt/here/the/path/to/your/dem/" # You need to obviously adapt that path to your case
dem_name = "whatever_name.tif" # You also need to adapt that file name... 
## Now we can load the dem into LSDTopytools: 
### already_preprocessed can be turn to True if you are 100% sure that your dem does not need preprocessing before flow routines
mydem = LSDDEM(path = path_to_dem, file_name = dem_name, already_preprocessed = False)

# Alright the dem is in the system and now needs to be preprocessed (if not done yet)
mydem.PreProcessing(filling = True, carving = True, minimum_slope_for_filling = 0.0001) # Unecessary if already preprocessed of course.

#Need to pregenerate a number of routines, it calculates flow direction, flow accumulation, drainage area , ...
mydem.CommonFlowRoutines()

# This define the river network, it is required to actually calculate other metrics
mydem.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = 1500)

# Defining catchment of interest: it extracts the catchments by outlet coordinates. You also need to adpat these obviously!!
## they need to be in the same coordinate system than the raster.
mydem.DefineCatchment( method="from_XY", X_coords = [532297,521028], Y_coords = [6188085,6196305])

# Calculates chi coordinate with an according theta
mydem.GenerateChi(theta = 0.4, A_0 = 1)

# At that stage you can get the basic river characteristics as follow:
# my_rivers = mydem.df_base_river
# my_rivers.to_csv("name_of_my_file_containing_base_river.csv", index = False) #  This saves the base rivers to csv


##############################
# Find below the range of specific chi related analysis you can run with lsdtopytools
# Note that this script contains a list of analysis as exhaustive as possible, but you do not need to run all of them at the same time !!
# It is rather adviced to run them separately when possible
##############################


######## ksn Extraction
# For the catchments previously selected and chi calculated with an appropriate theta/A0
# this extract the Normalised channel steepness (if A0=1) or the steepness of chi-z profiles
# using https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013JF002981
# mydem.GenerateChi(theta = 0.4, A_0 = 1) # this is if you need to generate chi with another concavity
mydem.ksn_MuddEtAl2014(target_nodes=70, n_iterations=60, skip=1, nthreads = 1)
## Plotting the results (see API for full details)
quickplot_ksn_knickpoints.plot_ksn_map(mydem, ksn_colormod = "percentile")
# You can access the data with: mydem.df_ksn

####### knickpoint extraction (requires ksn to be calculated)
mydem.knickpoint_extraction(self,lambda_TVD = "auto", combining_window = 30, window_stepped = 80, n_std_dev = 7)
# there is no plotting function yet (work in progress), however you can get the data as follow:
kp = mydem.df_knickpoint
# You can also "trim" the dataset by selecting certain river keys, basin keys and max values for each type of knickpoints:
## Full details in the API
mydem.trim_knickpoint_dataset(method = "third_quartile", select_river = [], select_basin = [], trimmer = {})
# Then you can get a trimmed dataset:
kp_selected = mydem.df_knickpoint_selecao
###
# the ouputs are pandas dataframes and can be saved to csv for external processing. They all contain river points with X/Y/elevation and various metrics
# example:
# kp_selected.to_csv("knickpoints_trimmed.csv", index=False)