#!/usr/bin/env python
"""
Command-line tool to control the concavity constraining tools
Mudd et al., 2018
So far mostly testing purposes
B.G.
"""
from lsdtopytools import LSDDEM, raster_loader as rl # I am telling python I will need this module to run.
from lsdtopytools import argparser_debug as AGPD # I am telling python I will need this module to run.
from lsdtopytools import quickplot as qp, quickplot_movern as qmn # We will need the plotting routines
import numpy as np
from matplotlib import pyplot as plt
from lsdtopytools import quickplot_utilities as QU
import time as clock # Basic benchmarking
import sys # manage the argv
import pandas as pd

def PreProcess():
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["breach"] = False
	default_param["fill"] = False
	default_param["min_slope"] = 0.0001
	default_param["save_path"] = "./"

	default_param = AGPD.ingest_param(default_param, sys.argv)

	if(default_param["help"] or len(sys.argv)==1):
		print("""
			This command-line tool provides basic Preprocessing functions for a raster.
			Breaching is achieved running https://doi.org/10.1002/hyp.10648 algorithm (Lindsay2016), with an implementation from RICHDEM
			Filling is from Wang and liu 2006. I think MDH implemented the code.
			It also "clean" the raster and tidy the nodata in it.
			To use, run the script with relevant options, for example:
			-> filling and breaching
				lsdtt-depressions file=myraster.tif fill breach 0.00001 

			option available:
				file: name of the raster (file=name.tif)
				path: path to the file (default = current folder)
				breach: want to breach?
				fill: want to fill?
				min_slope: imply a min slope for filling

			""")
		quit()

	sevadir = default_param["save_path"]
	
	print("Welcome to the command-line tool to .")
	print("Let me first load the raster ...")
	mydem = LSDDEM(file_name = default_param["file"], path=default_param["path"], already_preprocessed = False, verbose = False)
	print("Got it. Now dealing with the depressions ...")
	mydem.PreProcessing(filling = default_param["fill"], carving = default_param["breach"], minimum_slope_for_filling = float(default_param["min_slope"])) # Unecessary if already preprocessed of course.

	mydem.save_dir

	print("Saving the raster! same name basically, but with _PP at the end")
	rl.save_raster(mydem.cppdem.get_PP_raster(),mydem.x_min,mydem.x_max,mydem.y_max,mydem.y_min,mydem.resolution,mydem.crs,mydem.path+sevadir+mydem.prefix + "_PP.tif", fmt = 'GTIFF')

def Polyfit_Metrics():

	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["window_radius"] = 30

	default_param["average_elevation"] = False
	default_param["slope"] = False
	default_param["aspect"] = False
	default_param["curvature"] = False
	default_param["planform_curvature"] = False
	default_param["profile_curvature"] = False
	default_param["tangential_curvature"] = False
	default_param["TSP"] = False



	default_param = AGPD.ingest_param(default_param, sys.argv)

	if(default_param["help"] or len(sys.argv)==1):
		print("""
			Get the polyfit metrics: average_elevation, slope, aspect, curvature, planform_curvature, profile_curvature, tangential_curvature
			by fitting a plane equation for each pixels through the neighboring node within a distance.
			
			Quick example: lsdtt-polyfits file=myraster.tif slope window_radius=17
			option available:
				file: name of the raster (file=name.tif)
				path: path to the file (default = current folder)
				average_elevation(add keyword to activate): get the average_elevation of the polyfit raster
				slope(add keyword to activate): get the slope of the polyfit raster
				aspect(add keyword to activate): get the aspect of the polyfit raster
				curvature(add keyword to activate): get the curvature of the polyfit raster
				planform_curvature(add keyword to activate): get the planform_curvature of the polyfit raster
				profile_curvature(add keyword to activate): get the profile_curvature of the polyfit raster
				tangential_curvature(add keyword to activate): get the tangential_curvature of the polyfit raster
				window_radius: distance around each pixels to fit the plane

			Future improvements:
			- Automatic plotting option
			- others?
			""")
		quit()


	print("Welcome to the command-line tool to .")
	print("Let me first load the raster ...")
	mydem = LSDDEM(file_name = default_param["file"], path=default_param["path"], already_preprocessed = True, verbose = False)

	res = mydem.get_polyfit_metrics(window_radius = default_param["window_radius"], average_elevation = default_param["average_elevation"], slope = default_param["slope"], aspect = default_param["aspect"], curvature = default_param["curvature"], planform_curvature = default_param["planform_curvature"], profile_curvature = default_param["profile_curvature"], tangential_curvature = default_param["tangential_curvature"], TSP = default_param["TSP"], save_to_rast = True)

	if(default_param["slope"]):
		fig,ax = plt.subplots()
		cb =ax.imshow(res['slope'], extent = mydem.extent, vmin = np.percentile(res['slope'], 20), vmax = np.percentile(res['slope'], 80) )
		QU.fix_map_axis_to_kms(ax, 8,4)
		plt.colorbar(cb)
		plt.savefig(mydem.path+mydem.prefix + "_slope_%s.png"%(default_param["window_radius"]), dpi = 500)




def topomap():
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["save_hillshade"] = True
	default_param["topomap"] = True

	default_param = AGPD.ingest_param(default_param, sys.argv)

	if(default_param["help"] or len(sys.argv)==1):
		print("""
			This command-line tool provides first order topographic visualisation:

			option available:
				file: name of the raster (file=name.tif)
				path: path to the file (default = current folder)
				save_hillshade: save a tif hillshade file
				topomap: save a nice topographic map

			""")
		quit()


	print("Welcome to the command-line tool to .")
	print("Let me first load the raster ...")
	mydem = LSDDEM(file_name = default_param["file"], path=default_param["path"], already_preprocessed = True, verbose = False)

	if(default_param["topomap"]):
		qp.plot_nice_topography(mydem ,figure_width = 6, figure_width_units = "inches", cmap = "gist_earth", hillshade = True, 
	alpha_hillshade = 0.45, color_min = None, color_max = None, dpi = 500, output = "save", format_figure = "png", fontsize_ticks =6, fontsize_label = 8, 
	hillshade_cmin = 0, hillshade_cmax = 250,colorbar = True, 
	fig = None, ax = None, colorbar_label = None, colorbar_ax = None)

	if(default_param["save_hillshade"]):
		print("Saving the Hillshade raster ...")
		rl.save_raster(mydem.get_hillshade(),mydem.x_min,mydem.x_max,mydem.y_max,mydem.y_min,mydem.resolution,mydem.crs,mydem.path+mydem.prefix + "_hs.tif", fmt = 'GTIFF')

def remove_seas():
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["sea_level"] = 0
	default_param = AGPD.ingest_param(default_param, sys.argv)

	if(default_param["help"] or len(sys.argv)==1):
		print("""
			This command-line tool preprocess your raster to the right format and remove elevation below a threshold, typically the sea.

			option available:
				file: name of the raster (file=name.tif)
				path: path to the file (default = current folder)
				sea_level: elevation threshold (default = 0)

			""")
		quit()


	print("Loading the dem")
	mydem = LSDDEM(file_name = default_param["file"], path=default_param["path"], already_preprocessed = True, verbose = False)
	print("Processing the sea, removing everything %s ..."%(float(default_param["sea_level"])))
	topo = np.copy(mydem.cppdem.get_PP_raster())
	topo[ topo < float(default_param["sea_level"]) ] = -9999
	print("done, let me save the raster")
	rl.save_raster(topo,mydem.x_min,mydem.x_max,mydem.y_max,mydem.y_min,mydem.resolution,mydem.crs,mydem.path+mydem.prefix + "_sea_removed.tif", fmt = 'GTIFF')


def extract_single_river_from_source():
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["already_preprocessed"] = False
	default_param["X"] = 0
	default_param["Y"] = 0
	default_param["output_name"] = "Single_River"
	default_param = AGPD.ingest_param(default_param, sys.argv)

	if(default_param["help"] or len(sys.argv)==1):
		print("""
			This command-line tool preprocess your raster to the right format and remove elevation below a threshold, typically the sea.

			Quick example: lsdtt-extract-single-river file=test_raster.tif X=592149.7 Y=4103817.2

			Option available:
				file: name of the raster (file=name.tif)
				path: path to the file (default = current folder)
				already_preprocessed (default = False): Add the keyword if your raster is already preprocessed for flow analysis, False if it needs preprocessing. To get more control on the preprocessing you can use lsdtt-depressions command.
				X: Easting/X coordinate of the source
				Y: Easting/Y coordinate of the source
				output_name (default = Single_River): prefix of the ouput(s) files.

			Future improvements:
				- Automatic plotting of base statistic
				- Adding Chi/ksn/gradient options
			""")
		quit()

	print("This command-line tool extract a single river from a DEM from the XY coordinates of the source")
	print("Loading the dem")
	mydem = LSDDEM(file_name = default_param["file"], path=default_param["path"], already_preprocessed = bool(default_param["already_preprocessed"]), verbose = True)
	if(bool(default_param["already_preprocessed"]) == False):
		print("I am preprocessing your dem (carving + filling), if you have already done that, you can save time by adding the option already_preprocessed=True to the command line (it saves time)")
		mydem.PreProcessing()

	river = pd.DataFrame(mydem.ExtractSingleRiverFromSource(float(default_param["X"]),float(default_param["Y"])))
	river.to_csv("%s.csv"%(default_param["output_name"]), index = False)


# def basinator():
	