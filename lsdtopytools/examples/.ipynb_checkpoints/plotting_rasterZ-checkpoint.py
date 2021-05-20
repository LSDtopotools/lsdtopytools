import lsdtopytools as lsd
from matplotlib import pyplot as plt
import numpy as np
# Load mah raster
mydem = lsd.LSDDEM(file_name = "Switz_IsolateFixedChannel.bil", path = "U:/Datastore/CSCE/geos/groups/LSDTopoData/Switzerland/Emma_data/DEM/", already_preprocessed = True)

# # Create the figure and everything
fig,ax = lsd.quickplot.get_basemap(mydem , figsize = (4,5), cmap = "gist_earth", hillshade = True, 
	alpha_hillshade = 1, cmin = None, cmax = None,
	hillshade_cmin = -0.5, hillshade_cmax = 1, colorbar = False, 
	fig = None, ax = None, colorbar_label = None, colorbar_ax = None, fontsize_ticks = 8)

# Now you can plot any data in the same xy system on that shit
# e.g Add a raster on the top of it (generic, not necessarily a dem)

my_second_raster = lsd.raster_loader.load_raster("U:/Datastore/CSCE/geos/groups/LSDTopoData/Switzerland/Emma_data/DEM/Switz_KRaster.bil")
cb = ax.imshow(my_second_raster["array"], extent = my_second_raster["extent"], cmap = "coolwarm", zorder = 3, alpha = 0.6)
cbar = plt.colorbar(cb)
cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))



xlim = ax.get_xlim()
ylim = ax.get_ylim()
# If you have a csv:
#df = pd.read_csv("djsdg.csv")

# fig,ax = plt.subplots()

# Manually add polygon shapefile for it (I'll automate that at some points)
import geopandas as gpd
shp = gpd.read_file("U:/Datastore/CSCE/geos/groups/LSDTopoData/Switzerland/Emma_data/DEM/site_perimeter_utm.shp")

for idx in range(shp.shape[0]):

	# g = [i for i in ]

	all_coords = list(shp.geometry.exterior[idx].coords)            

	all_coords = np.array(all_coords)
	Xs = all_coords[:,0]
	Ys = all_coords[:,1]
	print(Xs)
	print(Ys)
	ax.fill(Xs,Ys, edgecolor = "#FFFB00", facecolor = "none", lw = 1.5, ls = "-.", zorder = 5)
	# ax.scatter(Xs,Ys, lw = 1, zorder = 5)


dem_array = mydem.cppdem.get_PP_raster()
dem_array[dem_array != -9999] = np.nan
ax.imshow(dem_array, extent = my_second_raster["extent"], cmap = "gray", vmin = -1000000000, vmax = -5656565, zorder = 4)

ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.show()
