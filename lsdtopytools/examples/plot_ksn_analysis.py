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
from lsdtopytools import quickplot, quickplot_ksn_knickpoints # We will need the plotting routines
import time as clock # Basic benchmarking
import sys # manage the argv

# Run with download to download the test dem:
# `python plot_ksn_analysis.py download` instead of `python plot_ksn_analysis.py`

##################################################################################################
# The following code download a test site in scotland. Replace it with your own raster if you need
# Requires wget, a small python portage of linux command wget to all OSs
# "pip install wget" will install it easily
if("download" in sys.argv):
	import wget
	print("Downloading a test dataset: ")
	file = wget.download("https://github.com/LSDtopotools/LSDTT_workshop_data/raw/master/WAWater.bil")
	wget.download("https://github.com/LSDtopotools/LSDTT_workshop_data/raw/master/WAWater.hdr")
	print("Done")
##################################################################################################

my_raster_path = "./" # I am telling saving my path to a variable. In this case, I assume the rasters is in the same folder than my script
file_name = "WAWater.bil" # The name of your raster with extension. RasterIO then takes care internally of decoding it. Good guy rasterio!

mydem = LSDDEM(path = my_raster_path, file_name = file_name) # If your dem is already preprocessed: filled or carved basically, add: , is_preprocessed = True
mydem.PreProcessing(filling = True, carving = True, minimum_slope_for_filling = 0.0001) # Unecessary if already preprocessed of course.
mydem.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = 1500)
mydem.GenerateChi(theta = 0.4)
mydem.DefineCatchment( method="from_XY", X_coords = [532297,521028], Y_coords = [6188085,6196305], test_edges = False)
mydem.ksn_MuddEtAl2014(target_nodes=70, n_iterations=60, skip=1, nthreads = 1)
quickplot_ksn_knickpoints.plot_ksn_map(mydem, ksn_colormod = "percentile")

print("Finished")
