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

def hypsometry_tools_general():
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["hypsometry"] = False
	default_param["absolute_elevation"] = False
	default_param["normalise_to_outlets"] = False
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
			This command-line tool run concavity analysis tools from LSDTopoTools.
			Description of the algorithms in Mudd et al., 2018 -> https://www.earth-surf-dynam.net/6/505/2018/
			To use, run the script with relevant options, for example:
				lsdtt-concavity-tools.py file=myraster.tif quick_movern X=5432 Y=78546

			option available:
				file: name of the raster (file=name.tif)
				path: path to the file (default = current folder)
				hypsometry: run hypsometric calculations
				absolute_elevation: hypsometric elevation will be the absolute value rather than the relavie/normalised one
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
	mydem.GenerateChi()

	print("Producing a figure of your catchement just for you to check if their location is OK!")
	qp.plot_check_catchments(mydem)


	if default_param["hypsometry"]:
		print("I am going to plot relative hypsometric curves for your different basins")
		print("This is quite quickly coded (for Jorien) and will probably evolve in the future!")
		
		fig,ax = plt.subplots()

		# Basin Array
		BAS = mydem.cppdem.get_chi_basin()
		TOPT = mydem.cppdem.get_PP_raster()
		DAD = mydem.cppdem.get_DA_raster()


		for bas in np.unique(BAS):
			if(bas != -9999):
				# getting hte topo raster ONLY for current basin
				this_topo = TOPT[BAS == bas].ravel()
				# Now getting the DA
				this_DA = DAD[BAS == bas].ravel()
				A = np.nanmax(this_DA) # maximum area of that basin

				# Normalising the topo
				if(default_param["absolute_elevation"] == False):
					this_topo = (this_topo - np.nanmin(this_topo))
					this_topo = this_topo/np.nanmax(this_topo)
				elif(default_param["normalise_to_outlets"]):
					this_topo = (this_topo - np.nanmin(this_topo))

				# Y is h/H, X is a/A
				Y =[];X=[]
				for i in range(101):
					if(default_param["absolute_elevation"] == False):
						Y.append(i*0.01)
					else:
						Y.append(np.percentile( this_topo,i))

					X.append(np.nanmax(this_DA[this_topo>=Y[-1]])/A)
				ax.plot(X,Y,lw = 1, label = bas, alpha = 0.8)

		ax.set_xlabel(r"$\frac{a}{A}$")
		if(default_param["absolute_elevation"] == False):
			ax.set_ylabel(r"$\frac{h}{H}$")
		else:
			ax.set_ylabel("Elevation (m)")

		ax.legend()

		if(default_param["absolute_elevation"] == False):
			suffix ="_relative_hypsometry"
		else:
			suffix = "_absolute_hypsometry"
			if(default_param["normalise_to_outlets"]):
				suffix +="_norm2outlet"

		plt.savefig(mydem.save_dir+mydem.prefix+suffix+"."+"png", dpi = 500)
		
		plt.clf()




	print("Finished!")

