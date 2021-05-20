#!/usr/bin/env python
"""
Command-line tool to control the concavity constraining tools
Mudd et al., 2018
So far mostly testing purposes
B.G.
"""
import matplotlib;matplotlib.use("Agg")
from lsdtopytools import LSDDEM # I am telling python I will need this module to run.
from lsdtopytools import argparser_debug as AGPD # I am telling python I will need this module to run.
from lsdtopytools import quickplot as qp, quickplot_movern as qmn # We will need the plotting routines
import time as clock # Basic benchmarking
import sys # manage the argv
import numpy as np
from matplotlib import pyplot as plt

def chi_mapping_tools():
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["map"] = False
	default_param["theta_ref"] = 0.45
	default_param["A_0"] = 1
	default_param["area_threshold"] = 500

	default_param["X"] = None
	default_param["Y"] = None

	default_param = AGPD.ingest_param(default_param, sys.argv)

	# Checking the param
	if(isinstance(default_param["X"],str)):
		X = [float(default_param["X"])]
		Y = [float(default_param["Y"])]
	else:
		X = [float(i) for i in default_param["X"]]
		Y = [float(i) for i in default_param["Y"]]


	if(default_param["help"] or len(sys.argv)==1):
		print("""
			This command-line tool run chi analysis tools from LSDTopoTools. This tool is focused on extracting chi, ksn or things like that.
			Chi -> DOI: 10.1002/esp.3302 An integral approach to bedrock river profile analysis, Perron and Royden 2013
			ksn -> (calculated from a robust statistical method based on chi): https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2013JF002981, Mudd et al., 2014

			
			To use, run the script with relevant options, for example:
				lsdtt-chi-tools file=myraster.tif map X=531586 Y=6189787 theta_ref=0.55 A_0=10

			option available:
				file: name of the raster (file=name.tif)
				path: path to the file (default = current folder)
				map: generate map of chi
				theta_ref: theta ref on the main equation
				A_0: A_0 on the chi equation
				area_threshold: the threshold of pixel to initiate the river network
				X: X coordinate of the outlet (So far only single basin is supported as this is an alpha tool)
				Y: Y coordinate of the outlet (So far only single basin is supported as this is an alpha tool)
				help: if written, diplay this message. Documentation soon to be written.

			""")

		quit()


	print("Welcome to the command-line tool to plot basin-wide hypsometry stuff!")
	print("Let me first load the raster ...")
	mydem = LSDDEM(file_name = default_param["file"], path=default_param["path"], already_preprocessed = False, verbose = False)
	print("Got it. Now dealing with the depressions ...")
	mydem.PreProcessing(filling = True, carving = True, minimum_slope_for_filling = 0.0001) # Unecessary if already preprocessed of course.
	print("Done! Extracting the river network")
	mydem.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = 1000)
	print("I have some rivers for you! Defining the watershed of interest...")


	mydem.DefineCatchment( method="from_XY", X_coords = X, Y_coords = Y, test_edges = False, coord_search_radius_nodes = 25, coord_threshold_stream_order = 1)
	print("I should have it now.")
	print("I got all the common info, now I am running what you want!")
	mydem.GenerateChi(theta = float(default_param["theta_ref"]) , A_0 = float(default_param["A_0"]))

	print("Producing a figure of your catchement just for you to check if their location is OK!")
	qp.plot_check_catchments(mydem)


	if default_param["map"]:
		print("I am going to plot chi maps")
		fig,ax = qp.plot_nice_topography(mydem ,figure_width = 4, figure_width_units = "inches", cmap = "gist_earth", hillshade = True, 
			alpha_hillshade = 1, color_min = None, color_max = None, dpi = 300, output = "return", format_figure = "png", fontsize_ticks =6, fontsize_label = 8, 
			hillshade_cmin = 0, hillshade_cmax = 250,colorbar = False, colorbar_label = None, colorbar_ax = None)


		normalize = matplotlib.colors.Normalize(vmin = mydem.df_base_river["drainage_area"].min(), vmax = mydem.df_base_river["drainage_area"].max())
		mydem.df_base_river = mydem.df_base_river[mydem.df_base_river["chi"]>=0]
		cb =ax.scatter(mydem.df_base_river["x"],mydem.df_base_river["y"], s = 4*normalize(mydem.df_base_river["drainage_area"].values), c = mydem.df_base_river["chi"], vmin = mydem.df_base_river["chi"].quantile(0.1), vmax = mydem.df_base_river["chi"].quantile(0.9), zorder = 5, lw = 0)
		plt.colorbar(cb)
		plt.savefig(mydem.save_dir+mydem.prefix+"_chi_map"+"."+"png", dpi = 500)
		plt.clf()




	print("Finished!")

