"""
This example script download a test raster, caculates and plot normalised channel steepness (ksn).
Read the comments to understand each steps. Copy and adapt this script to learn.
If any questions: b.gailleton@sms.ed.ac.uk
B.G.
"""
# If you are facing a common matplotlib issue, uncomment that:
#################################################
# import matplotlib
# matplotlib.use("Agg")
#################################################
from lsdtopytools import LSDDEM # I am telling python I will need this module to run.
from lsdtopytools import quickplot, quickplot_movern # We will need the plotting routines
import time as clock # Basic benchmarking
import sys # manage the argv
from matplotlib import pyplot as plt # plotting
import numpy as np # 

# Run with download to download the test dem:
# `python plot_ksn_analysis.py download` instead of `python plot_ksn_analysis.py`

##################################################################################################
# The following code download a test site in scotland. Replace it with your own raster if you need
# Requires wget, a small python portage of linux command wget to all OSs
# "pip install wget" will install it easily
if(len(sys.argv)>1):
	if(sys.argv[1].lower() == "download"):
		import wget
		print("Downloading a test dataset: ")
		file = wget.download("https://github.com/LSDtopotools/LSDTT_workshop_data/raw/master/WAWater.bil")
		wget.download("https://github.com/LSDtopotools/LSDTT_workshop_data/raw/master/WAWater.hdr")
		print("Done")
##################################################################################################


my_raster_path = "./" # I am telling saving my path to a variable. In this case, I assume the rasters is in the same folder than my script
file_name = "WAWater.bil" # The name of your raster with extension. RasterIO then takes care internally of decoding it. Good guy rasterio!

# I am now telling lsdtopytools where is my raster, and What do I want to do with it. No worries It will deal with the detail internally
mydem = LSDDEM(path = my_raster_path, file_name = file_name) # If your dem is already preprocessed: filled or carved basically, add: , is_preprocessed = True
## Loaded in the system, now preprocessing: I want to carve it and imposing a minimal slope on remaining flat surfaces: 0.0001
mydem.PreProcessing(filling = True, carving = True, minimum_slope_for_filling = 0.0001) # Unecessary if already preprocessed of course.

mydem.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = 500)
mydem.DefineCatchment( method="from_XY", X_coords = [527107, 527033, 530832], Y_coords = [6190656, 6191745, 6191015])
mydem.GenerateChi(theta=0.45,A_0 = 1)

print("Starting movern extraction")
mydem.cppdem.calculate_movern_disorder(0.1, 0.05, 18, 1, 1000) # start theta, delta, n, A0, threashold
print("movern done, getting the data")

quickplot_movern.plot_disorder_results(mydem, normalise = True, figsize = (4,3), dpi = 300, output = "save", format_figure = "png", legend = True,
 cumulative_best_fit = True)
quickplot_movern.plot_disorder_map(mydem ,figure_width = 4, figure_width_units = "inches", cmap = "jet", alpha_hillshade = 0.95, 
	this_fontsize = 6, alpha_catchments = 0.75,  dpi = 300, output = "save", format_figure = "png")