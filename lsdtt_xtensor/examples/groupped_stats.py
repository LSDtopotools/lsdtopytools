# Takes 2 rasters: the first contains integer values defining groups of data (e.g., a rasterized shapefile)
# the second one contains values to group (e.g., slopes or elevation)
from lsdtt_xtensor_python import comparison_stats_from_2darrays as gpc
import numpy as np
# from matplotlib import pyplot as plt

#TODO add real data here

# First array is the index one
index_array = np.ones((5000,5000), dtype = np.int32)
index_array[0:200,:] = 5
index_array[200:300,:] = 4
index_array[300:400,:] = 1

# Second array contains values
val_array = np.random.rand(5000,5000).astype(np.float32)

print("Getting values in CPP")
# The code takes the following arguments: the 2 arrays ofc, an optional value to ignore (e.g., NoDataValue), and the number of rows and cols
test = gpc(index_array,val_array,10000000,index_array.shape[0],index_array.shape[1])

print("Done")

