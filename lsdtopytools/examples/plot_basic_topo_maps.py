"""
This example script download a test raster, and plot different first order figure, for example nice topographic map.
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
from lsdtopytools import quickplot # We will need the plotting routines

##################################################################################################
# The following code download a test site in scotland. Replace it with your own raster if you need
# Requires wget, a small python portage of linux command wget to all OSs
# "pip install wget" will install it easily
# import wget
# print("Downloading a test dataset: ")
# file = wget.download("https://github.com/LSDtopotools/LSDTT_workshop_data/raw/master/WAWater.bil")
# wget.download("https://github.com/LSDtopotools/LSDTT_workshop_data/raw/master/WAWater.hdr")
# print("Done")
##################################################################################################

my_raster_path = "./" # I am telling saving my path to a variable. In this case, I assume the rasters is in the same folder than my script
file_name = "WAWater.bil" # The name of your raster with extension. RasterIO then takes care internally of decoding it. Good guy rasterio!

# I am now telling lsdtopytools where is my raster, and What do I want to do with it. No worries It will deal with the detail internally
mydem = LSDDEM(path = my_raster_path, file_name = file_name) # If your dem is already preprocessed: filled or carved basically, add: , is_preprocessed = True
## Loaded in the system, now preprocessing: I want to carve it and imposing a minimal slope on remaining flat surfaces: 0.0001
mydem.PreProcessing(filling = True, carving = True, minimum_slope_for_filling = 0.0001) # Unecessary if already preprocessed of course.
## First plotting a nice topography
quickplot.plot_nice_topography(mydem) # many available options are available to custumize, see documentation.

## Now plotting the pre-processing differential raster to check the results and potential artifacts
quickplot.plot_preprocessing_diff(mydem) # many available options are available to custumize, see documentation.

# Now I am testing a fill-only preprocessing to check the difference
mydem.PreProcessing(filling = True, carving = False, minimum_slope_for_filling = 0.0001) # Unecessary if already preprocessed of course.
quickplot.plot_preprocessing_diff(mydem) # many available options are available to custumize, see documentation.

print("Finished")
