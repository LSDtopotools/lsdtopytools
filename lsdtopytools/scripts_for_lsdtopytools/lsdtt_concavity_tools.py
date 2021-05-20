#!/usr/bin/env python
"""
Command-line tool to control the concavity constraining tools
Mudd et al., 2018
So far mostly testing purposes
B.G.
"""
from lsdtopytools import LSDDEM # I am telling python I will need this module to run.
from lsdtopytools import argparser_debug as AGPD # I am telling python I will need this module to run.
from lsdtopytools import quickplot as qp, quickplot_movern as qmn # We will need the plotting routines
import time as clock # Basic benchmarking
import sys # manage the argv
import pandas as pd
import lsdtopytools as lsd
import numpy as np
import numba as nb
import pandas as pd
from matplotlib import pyplot as plt
import sys
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import math
from lsdtopytools.numba_tools import travelling_salesman_algortihm, remove_outliers_in_drainage_divide
import random
import matplotlib.gridspec as gridspec
from multiprocessing import Pool, current_process
from scipy import spatial

class NoBasinFoundError(Exception):
	pass

def main_concavity():
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["quick_movern"] = False
	default_param["X"] = None
	default_param["Y"] = None

	default_param = AGPD.ingest_param(default_param, sys.argv)


	# Checking the param
	if(isinstance(default_param["X"],str)):
		X = [float(default_param["X"])]
		Y = [float(default_param["Y"])]
	else:
		try:
			X = [float(i) for i in default_param["X"]]
			Y = [float(i) for i in default_param["Y"]]
		except:
			pass

	if(default_param["help"] or len(sys.argv)==2 or "help" in sys.argv ):
		print("""
			This command-line tool run concavity analysis tools from LSDTopoTools.
			Description of the algorithms in Mudd et al., 2018 -> https://www.earth-surf-dynam.net/6/505/2018/
			To use, run the script with relevant options, for example:
				lsdtt-concavity-tools.py file=myraster.tif quick_movern X=5432 Y=78546

			option available:
				file: name of the raster (file=name.tif)
				path: path to the file (default = current folder)
				quick_movern: run disorder metrics and plot a result figure (you jsut need to write it)
				X: X coordinate of the outlet (So far only single basin is supported as this is an alpha tool)
				Y: Y coordinate of the outlet (So far only single basin is supported as this is an alpha tool)
				help: if written, diplay this message. Documentation soon to be written.
			""")
		quit()


	print("Welcome to the command-line tool to constrain your river network concavity. Refer to Mudd et al., 2018 -> https://www.earth-surf-dynam.net/6/505/2018/ for details about these algorithms.")
	print("Let me first load the raster ...")
	try:
		mydem = LSDDEM(file_name = default_param["file"], path=default_param["path"], already_preprocessed = False, verbose = False)
	except:
		print("Testing data still to build")
	print("Got it. Now dealing with the depressions ...")
	mydem.PreProcessing(filling = True, carving = True, minimum_slope_for_filling = 0.0001) # Unecessary if already preprocessed of course.
	print("Done! Extracting the river network")
	mydem.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = 1000)



	print("I have some rivers for you! Defining the watershed of interest...")
	mydem.DefineCatchment( method="from_XY", X_coords = X, Y_coords = Y, test_edges = False, coord_search_radius_nodes = 25, coord_threshold_stream_order = 1)
	print("I should have it now.")
	print("I got all the common info, now I am running what you want!")

	if(default_param["quick_movern"]):
		print("Initialising Chi-culations (lol)")
		mydem.GenerateChi()
		print("Alright, getting the disorder metrics for each chi values. LSDTopoTools can split a lot of messages, sns.")
		mydem.cppdem.calculate_movern_disorder(0.15, 0.05, 17, 1, 1000) # start theta, delta, n, A0, threashold
		print("I am done, plotting the results now")
		qmn.plot_disorder_results(mydem, legend = False, normalise = True, cumulative_best_fit = False)
		qp.plot_check_catchments(mydem)
		qmn.plot_disorder_map(mydem, cmap = "RdBu_r")
		print("FInished with quick disorder metric")

	print("Finished!")


def temp_concavity_FFS_all():
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["already_preprocessed"] = False
	default_param["process_main"] = False
	default_param["X"] = None
	default_param["Y"] = None
	default_param["AT"] = None
	default_param["ATM"] = None
	default_param["n_proc"] = None
	default_param["min_DA"] = None
	default_param["max_DA"] = None
	default_param["prefix"] = ""

	prefix = default_param["prefix"]

	default_param = AGPD.ingest_param(default_param, sys.argv)

	if(default_param["help"] or len(sys.argv)==1 or "help" in sys.argv):
		print("""
			Experimental command-line tools for concavity constraining. Deprecated now.
			""")
		quit()

	X = float(default_param["X"])
	Y = float(default_param["Y"])
	min_DA = float(default_param["min_DA"])
	max_DA = float(default_param["max_DA"])


	area_threshold_main_basin = float(default_param["ATM"])
	area_threshold = float(default_param["AT"])
	n_proc = int(default_param["n_proc"])

	dem_name = default_param["file"]
	if(default_param["process_main"]):
		lsd.concavity_automator.process_main_basin(dem_name, dem_path = "./" ,already_preprocessed = True , X_outlet = X, Y_outlet = Y, area_threshold = area_threshold, area_threshold_to_keep_river = area_threshold_main_basin, prefix = default_param["prefix"])
	else:
		lsd.concavity_automator.get_all_concavity_in_range_of_DA_from_baselevel(dem_name, dem_path = "./" ,already_preprocessed = True , X_outlet = X, Y_outlet = Y, min_DA = min_DA, max_DA = max_DA, area_threshold = area_threshold, n_proc = n_proc, prefix = default_param["prefix"])
		lsd.concavity_automator.post_process_basins(prefix = default_param["prefix"])
	
	for key,val in default_param.items():
		default_param[key] = [val]

	pd.DataFrame(default_param).to_csv(prefix +"log_concavity_FFS_sstuff.txt", index = False)

def concavity_single_basin():
	"""
Command-line tool to constrain concavity for a single basin from its XY coordinates using Normalised disorder method.
Takes several arguments (the values after = are example values to adapt): 
file=NameOfFile.tif -> The code NEEDS the neame of the raster to process.
already_preprocessed -> OPTIONAL Tell the code your raster does not need preprocessing, otherwise carve the DEM (see lsdtt-depressions for more options)
X=234 -> X Coordinate (in map unit) of the outlet (needs to be the exact pixel at the moment, will add a snapping option later) 
Y=234 -> Y Coordinate (in map unit) of the outlet (needs to be the exact pixel at the moment, will add a snapping option later) 
AT=5000 -> Area threshold in number of pixel to initiate a river: lower nummber <=> denser network (quickly increases the processing time)
prefix=test -> OPTIONAL Add a prefix to each outputted file (handy for automation)
mode=g -> DEFAULT is g . Processing mode: can be "g" for generating data, "p" or plotting previously generated data or "all" for everything.

Example:

lsdtt-concFFS-single file=DEM.tif already_preprocessed X=43422 Y=5353497 AT=2500 prefix=FeatherRiver mode=all
			"""
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["already_preprocessed"] = False
	default_param["X"] = None
	default_param["Y"] = None
	default_param["AT"] = None
	default_param["prefix"] = ""
	default_param["mode"] = "g"
	default_param["n_tribs_by_combinations"] = 3
	# Ingesting the parameters
	default_param = AGPD.ingest_param(default_param, sys.argv)

	prefix = default_param["prefix"]

	if(default_param["help"] or len(sys.argv)==1 or "help" in sys.argv):
		print("""
Command-line tool to constrain concavity for a single basin from its XY coordinates using Normalised disorder method.
Takes several arguments (the values after = are example values to adapt): 
file=NameOfFile.tif -> The code NEEDS the neame of the raster to process.
already_preprocessed -> OPTIONAL Tell the code your raster does not need preprocessing, otherwise carve the DEM (see lsdtt-depressions for more options)
X=234 -> X Coordinate (in map unit) of the outlet (needs to be the exact pixel at the moment, will add a snapping option later) 
Y=234 -> Y Coordinate (in map unit) of the outlet (needs to be the exact pixel at the moment, will add a snapping option later) 
AT=5000 -> Area threshold in number of pixel to initiate a river: lower nummber <=> denser network (quickly increases the processing time)
prefix=test -> OPTIONAL Add a prefix to each outputted file (handy for automation)
mode=g -> DEFAULT is g . Processing mode: can be "g" for generating data, "p" or plotting previously generated data or "all" for everything.

Example:

lsdtt-concFFS-single file=DEM.tif already_preprocessed X=43422 Y=5353497 AT=2500 prefix=FeatherRiver mode=all
			""")
		quit()

	# Reformatting some values that sometimes are not formatted correctly
	X = float(default_param["X"])
	Y = float(default_param["Y"])
	area_threshold = float(default_param["AT"])
	dem_name = default_param["file"]
	default_param["n_tribs_by_combinations"] = int(default_param["n_tribs_by_combinations"])
	# Wrapper to the processing function (the convoluted ls method makes multiprocessing easier when processing several basins)
	ls = [0,X,Y,area_threshold,prefix]
	# Calling th requested codes
	if("all" in default_param["mode"].lower() or "g" in default_param["mode"].lower()):
		lsd.concavity_automator.process_basin(ls,ignore_numbering = True, overwrite_dem_name = dem_name, n_tribs_by_combo =  default_param["n_tribs_by_combinations"])
	if("all" in default_param["mode"].lower() or "p" in default_param["mode"].lower()):
		lsd.concavity_automator.plot_basin(ls,ignore_numbering = True, overwrite_dem_name = dem_name)
	
	# Saving a log of processing
	for key,val in default_param.items():
		default_param[key] = [val]
	pd.DataFrame(default_param).to_csv(prefix +"log_concavity_FFS_single_basin_sstuff.txt", index = False)


def concavity_multiple_basin():
	"""
Command-line tool to constrain concavity for a multiple basin from their XY coordinates using Normalised disorder method.
Takes several arguments (the values after = are example values to adapt): 
file=NameOfFile.tif -> The code NEEDS the neame of the raster to process.
already_preprocessed -> OPTIONAL Tell the code your raster does not need preprocessing, otherwise carve the DEM (see lsdtt-depressions for more options)
csv=outlets.csv -> Name of the csv file containing the following columns: "X", "Y" and "area_threshold" for each basins to investigate. Can be generated automatically from lsdtt-concFFS-spawn-outlets
n_proc=4 -> DEFAULT is 1. Number of processors to use in parallel when possible.
prefix=test -> OPTIONAL Add a prefix to each outputted file (handy for automation)
mode=g -> DEFAULT is g . Processing mode: can be "g" for generating data, "p" or plotting previously generated data, "d" for plotting disorder map (WARNING takes time and memory) "all" for everything.

Example:

lsdtt-concFFS-multiple file=DEM.tif already_preprocessed csv=FeatherRiveroutlets.csv prefix=FeatherRiver mode=g
			"""
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["already_preprocessed"] = False
	default_param["prefix"] = ""
	default_param["csv"] = ""
	default_param["n_proc"] = 1
	default_param["area_thershold_basin_extraction"] = 500
	default_param["precipitation_file"] = ""
	default_param["mode"] = "g"
	default_param["n_tribs_by_combinations"] = 3
	default_param = AGPD.ingest_param(default_param, sys.argv)
	prefix = default_param["prefix"]

	if(default_param["help"] or len(sys.argv)==1 or "help" in sys.argv):
		print("""
Command-line tool to constrain concavity for a multiple basin from their XY coordinates using Normalised disorder method.
Takes several arguments (the values after = are example values to adapt): 
file=NameOfFile.tif -> The code NEEDS the neame of the raster to process.
already_preprocessed -> OPTIONAL Tell the code your raster does not need preprocessing, otherwise carve the DEM (see lsdtt-depressions for more options)
csv=outlets.csv -> Name of the csv file containing the following columns: "X", "Y" and "area_threshold" for each basins to investigate. Can be generated automatically from lsdtt-concFFS-spawn-outlets
n_proc=4 -> DEFAULT is 1. Number of processors to use in parallel when possible.
prefix=test -> OPTIONAL Add a prefix to each outputted file (handy for automation)
mode=g -> DEFAULT is g . Processing mode: can be "g" for generating data, "p" or plotting previously generated data, "all" for everything.

Example:

lsdtt-concFFS-multiple file=DEM.tif already_preprocessed csv=FeatherRiveroutlets.csv prefix=FeatherRiver mode=g
			""")

	# Reading the csv file 
	df = pd.read_csv(default_param["csv"])

	# Reformatting stuff
	n_proc = int(default_param["n_proc"])
	default_param["n_tribs_by_combinations"] = int(default_param["n_tribs_by_combinations"])
	area_thershold_basin_extraction = float(default_param["area_thershold_basin_extraction"])
	dem_name = default_param["file"]

	if(default_param["precipitation_file"] == ""):
		precipitation = False
	else:
		precipitation = True

	# Processing options
	if("all" in default_param["mode"].lower() or "g" in default_param["mode"].lower()):
		lsd.concavity_automator.process_multiple_basins(dem_name, dem_path = "./",already_preprocessed = default_param["already_preprocessed"], 
			prefix = default_param["prefix"], X_outlets = df["X"].values, Y_outlets = df["Y"].values, n_proc = n_proc, area_threshold = df["area_threshold"].values, 
			area_thershold_basin_extraction = area_thershold_basin_extraction, n_tribs_by_combo =  default_param["n_tribs_by_combinations"],
			 use_precipitation_raster = precipitation, name_precipitation_raster = default_param["precipitation_file"])
		lsd.concavity_automator.post_process_analysis_for_Dstar(default_param["prefix"], n_proc = n_proc, base_raster_full_name = dem_name)
	
	if("z" in default_param["mode"].lower()):
		lsd.concavity_automator.post_process_analysis_for_Dstar(default_param["prefix"], n_proc = n_proc, base_raster_full_name = dem_name)
	
	if("p" in default_param["mode"].lower()):
		lsd.concavity_automator.plot_main_figures(default_param["prefix"])
		

	if("all" in default_param["mode"].lower() or "d" in default_param["mode"].lower()):
		lsd.concavity_automator.plot_multiple_basins(dem_name, dem_path = "./",already_preprocessed = default_param["already_preprocessed"], 
			prefix = default_param["prefix"], X_outlets = df["X"].values, Y_outlets = df["Y"].values, n_proc = n_proc, area_threshold = df["area_threshold"].values, 
			area_thershold_basin_extraction = area_thershold_basin_extraction, plot_Dstar = False, n_tribs_by_combo =  default_param["n_tribs_by_combinations"])
		lsd.concavity_automator.plot_Dstar_maps_for_all_concavities(default_param["prefix"], n_proc = n_proc)


	# Saving logs
	for key,val in default_param.items():
		default_param[key] = [val]

	pd.DataFrame(default_param).to_csv(prefix +"log_concavity_FFS_multiple_basin_sstuff.txt", index = False)

def spawn_XY_outlet():
	"""
Command-line tool to prechoose the basins used for other analysis. Outputs a file with outlet coordinates readable from other command-line tools and a basin perimeter csv readable by GISs to if the basins corresponds to your needs.
Takes several arguments (the values after = are example values to adapt): 
file=NameOfFile.tif -> The code NEEDS the neame of the raster to process.
already_preprocessed -> OPTIONAL Tell the code your raster does not need preprocessing, otherwise carve the DEM (see lsdtt-depressions for more options)
test_edges -> OPTIONAL will test if the basin extracted are potentially influenced by nodata and threfore uncomplete. WARNING, will take out ANY basin potentially cut, if you know what you are doing, you can turn off.
prefix=test -> OPTIONAL Add a prefix to each outputted file (handy for automation)
method=from_range -> DEFAULT from_range: determine the method to select basin. Can be
		from_range -> select largest basins bigger than min_DA but smaller than max_DA (in m^2)
		min_area -> select largest basins bigger than min_DA
		main_basin -> select the largest basin
		Other methods to come.
min_elevation=45 -> DEFAULT 0. Ignore any basin bellow that elevation
area_threshold=3500 -> DEFAULT 5000. River network area threshold in number of pixels (part of the basin selection is based on river junctions HIGHLY sensitive to that variable).

Example:
lsdtt-concFFS-spawn-outlets file=DEM.tif already_preprocessed min_DA=1e7 max_DA=1e9 area_threshold=3500
			"""

	default_param = AGPD.get_common_default_param()
	default_param["already_preprocessed"] = False
	default_param["test_edges"] = False
	default_param["area_threshold"] = 5000
	default_param["method"] = "from_range"
	default_param["min_DA"] = 1e6
	default_param["max_DA"] = 1e9
	default_param["min_elevation"] = 0;
	default_param["prefix"] = "";
	default_param = AGPD.ingest_param(default_param, sys.argv)

	choice_of_method = ["min_area", "main_basin","from_range"]

	if(default_param["help"] or len(sys.argv)==1 or "help" in sys.argv):
		print("""
Command-line tool to prechoose the basins used for other analysis. Outputs a file with outlet coordinates readable from other command-line tools and a basin perimeter csv readable by GISs to if the basins corresponds to your needs.
Takes several arguments (the values after = are example values to adapt): 
file=NameOfFile.tif -> The code NEEDS the neame of the raster to process.
already_preprocessed -> OPTIONAL Tell the code your raster does not need preprocessing, otherwise carve the DEM (see lsdtt-depressions for more options)
test_edges -> OPTIONAL will test if the basin extracted are potentially influenced by nodata and threfore uncomplete. WARNING, will take out ANY basin potentially cut, if you know what you are doing, you can turn off.
prefix=test -> OPTIONAL Add a prefix to each outputted file (handy for automation)
method=from_range -> DEFAULT from_range: determine the method to select basin. Can be
		from_range -> select largest basins bigger than min_DA but smaller than max_DA (in m^2)
		min_area -> select largest basins bigger than min_DA
		main_basin -> select the largest basin
		Other methods to come.
min_elevation=45 -> DEFAULT 0. Ignore any basin bellow that elevation
area_threshold=3500 -> DEFAULT 5000. River network area threshold in number of pixels (part of the basin selection is based on river junctions HIGHLY sensitive to that variable).

Example:
lsdtt-concFFS-spawn-outlets file=DEM.tif already_preprocessed min_DA=1e7 max_DA=1e9 area_threshold=3500
			""")
		return 0;

	# Checks if the method requested is valid or not
	if(default_param["method"].lower() not in choice_of_method):
		print("I cannot recognise the method! Please choose from:")
		print(choice_of_method)
		return 0

	# Formatting parameters
	area_threshold = int(default_param["area_threshold"])
	min_DA = float(default_param["min_DA"])
	max_DA = float(default_param["max_DA"])
	min_elevation = float(default_param["min_elevation"])

	# Reading DEM
	mydem = LSDDEM(file_name = default_param["file"], path = default_param["path"], already_preprocessed = default_param["already_preprocessed"], remove_seas = True, sea_level = min_elevation)
	if(default_param["already_preprocessed"] == False):
		mydem.PreProcessing()
	# Extracting basins
	mydem.CommonFlowRoutines()
	print("Done with flow routines")
	mydem.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = area_threshold)
	# Get the outlet coordinates of all the extracted basins
	print("Extracted rivers")
	df_outlet = mydem.DefineCatchment(  method = default_param["method"], min_area = min_DA, max_area = max_DA, test_edges = default_param["test_edges"])#, X_coords = [X_coordinates_outlets[7]], Y_coords = [Y_coordinates_outlets[7]])
	print("Extracted")

	for key,val in df_outlet.items():
		df_outlet[key] = np.array(df_outlet[key])
	# Getting the rivers
	mydem.GenerateChi(theta = 0.4, A_0 = 1)
	# Saing the rivers to csv
	mydem.df_base_river.to_csv(default_param["prefix"]+"rivers.csv", index = False)
	#Saving the outlet
	df_outlet["area_threshold"] = np.full(df_outlet["X"].shape[0],area_threshold)

	# print(df_outlet)

	pd.DataFrame(df_outlet).to_csv(default_param["prefix"]+"outlets.csv", index = False)

	# Getting the perimeter of basins
	this = mydem.cppdem.extract_perimeter_of_basins()
	df_perimeter = {"X":[],"Y":[],"Z":[],"IDs":[]}
	for key,val in this.items():
		df_perimeter["X"].append(np.array(val["X"]))
		df_perimeter["Y"].append(np.array(val["Y"]))
		df_perimeter["Z"].append(np.array(val["Z"]))
		df_perimeter["IDs"].append(np.full(np.array(val["Z"]).shape[0], key))

	for key,val in df_perimeter.items():
		df_perimeter[key] = np.concatenate(val)



	pd.DataFrame(df_perimeter).to_csv(default_param["prefix"]+"perimeters.csv", index = False)

def spawn_XY_outlet_subbasins():

	default_param = AGPD.get_common_default_param()
	default_param["already_preprocessed"] = False
	default_param["X"] = 0
	default_param["Y"] = 0
	default_param["area_threshold"] = 5000
	default_param["min_DA"] = 1e6
	default_param["max_DA"] = 1e9
	default_param["min_elevation"] = 0;
	default_param["prefix"] = "";
	default_param = AGPD.ingest_param(default_param, sys.argv)

	if(default_param["help"] or len(sys.argv)==1 or "help" in sys.argv):
		print("""
Command-line tool to extract basin information about all the subbasins within a main one. Outputs a file with outlet coordinates readable from other command-line tools and a basin perimeter csv readable by GISs to if the basins corresponds to your needs.
Takes several arguments (the values after = are example values to adapt): 
file=NameOfFile.tif -> The code NEEDS the neame of the raster to process.
already_preprocessed -> OPTIONAL Tell the code your raster does not need preprocessing, otherwise carve the DEM (see lsdtt-depressions for more options)
prefix=test -> OPTIONAL Add a prefix to each outputted file (handy for automation)
min_elevation=45 -> DEFAULT 0. Ignore any basin bellow that elevation
area_threshold=3500 -> DEFAULT 5000. River network area threshold in number of pixels (part of the basin selection is based on river junctions HIGHLY sensitive to that variable).
min_DA=1e7 -> minimum drainage area to extract a subbasin
max_DA=1e9 -> maximum drainage area for a subbasin
X=234 -> X Coordinate (in map unit) of the outlet (needs to be the exact pixel at the moment, will add a snapping option later) 
Y=234 -> Y Coordinate (in map unit) of the outlet (needs to be the exact pixel at the moment, will add a snapping option later) 

Example:
lsdtt-concFFS-spawn-outlets file=DEM.tif already_preprocessed min_DA=1e7 max_DA=1e9 area_threshold=3500
			""")
		return 0;

	area_threshold = int(default_param["area_threshold"])
	X = float(default_param["X"])
	min_DA = float(default_param["min_DA"])
	Y = float(default_param["Y"])
	max_DA = float(default_param["max_DA"])
	min_elevation = float(default_param["min_elevation"])

	mydem = LSDDEM(file_name = default_param["file"], path = default_param["path"], already_preprocessed = default_param["already_preprocessed"], remove_seas = True, sea_level = min_elevation)
	if(default_param["already_preprocessed"] == False):
		mydem.PreProcessing()
	# Extracting basins
	mydem.CommonFlowRoutines()
	mydem.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = area_threshold)
	# df_outlet = mydem.DefineCatchment(  method = default_param["method"], min_area = min_DA, max_area = max_DA, test_edges = default_param["test_edges"])#, X_coords = [X_coordinates_outlets[7]], Y_coords = [Y_coordinates_outlets[7]])
	df_outlet = mydem.cppdem.calculate_outlets_min_max_draining_to_baselevel(X, Y, min_DA, max_DA,500)
	mydem.check_catchment_defined = True


	for key,val in df_outlet.items():
		df_outlet[key] = np.array(df_outlet[key])
	mydem.GenerateChi(theta = 0.4, A_0 = 1)
	mydem.df_base_river.to_csv(default_param["prefix"]+"rivers.csv", index = False)
	df_outlet["area_threshold"] = np.full(df_outlet["X"].shape[0],area_threshold)
	df_outlet = pd.DataFrame(df_outlet)
	df_outlet.to_csv(default_param["prefix"]+"outlets.csv", index = False)
	df_outlet["ID"] = np.array(list(range(df_outlet.shape[0])))	

	this = mydem.cppdem.extract_perimeter_of_basins()
	df_perimeter = {"X":[],"Y":[],"Z":[],"IDs":[]}
	for key,val in this.items():
		df_perimeter["X"].append(np.array(val["X"]))
		df_perimeter["Y"].append(np.array(val["Y"]))
		df_perimeter["Z"].append(np.array(val["Z"]))
		df_perimeter["IDs"].append(np.full(np.array(val["Z"]).shape[0], key))


	## Log from the analysis
	for key,val in df_perimeter.items():
		df_perimeter[key] = np.concatenate(val)
	pd.DataFrame(df_perimeter).to_csv(default_param["prefix"]+"perimeters.csv", index = False)

	

def temp_concavity_FFS_all_test_func(default_param):
	# Here are the different parameters and their default value fr this script
	# default_param = AGPD.get_common_default_param()
	# default_param["already_preprocessed"] = False
	# default_param["X"] = None
	# default_param["Y"] = None
	# default_param["AT"] = None
	# default_param["ATM"] = None
	# default_param["n_proc"] = None
	# default_param["min_DA"] = None
	# default_param["max_DA"] = None
	# default_param["prefix"] = ""

	# default_param = AGPD.ingest_param(default_param, sys.argv)


	X = float(default_param["X"])
	Y = float(default_param["Y"])
	min_DA = float(default_param["min_DA"])
	max_DA = float(default_param["max_DA"])


	area_threshold_main_basin = float(default_param["ATM"])
	area_threshold = float(default_param["AT"])
	n_proc = int(default_param["n_proc"])

	dem_name = default_param["file"]

	lsd.concavity_automator.get_all_concavity_in_range_of_DA_from_baselevel(dem_name, dem_path = "./" ,already_preprocessed = True , X_outlet = X, Y_outlet = Y, min_DA = min_DA, max_DA = max_DA, area_threshold = area_threshold, n_proc = n_proc, prefix = default_param["prefix"])
	lsd.concavity_automator.post_process_basins(prefix = default_param["prefix"])
	lsd.concavity_automator.process_main_basin(dem_name, dem_path = "./" ,already_preprocessed = True , X_outlet = X, Y_outlet = Y, area_threshold = area_threshold, area_threshold_to_keep_river = area_threshold_main_basin, prefix = default_param["prefix"])

	for key,val in default_param.items():
		default_param[key] = [val]

	pd.DataFrame(default_param).to_csv(default_param["prefix"] +"log_concavity_FFS_sstuff.txt", index = False)



def concavity_FFS_down_to_top():
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["already_preprocessed"] = False
	default_param["save_XY_array"] = False
	default_param["X_source"] = None
	default_param["Y_source"] = None
	default_param["min_elevation"] = None
	default_param["area_threshold"] = None
	default_param["flow_distance_step"] = None
	default_param["min_DA"] = None
	default_param["prefix"] = ""
	default_param = AGPD.ingest_param(default_param, sys.argv)


	if(default_param["help"] or len(sys.argv)==1 or "help" in sys.argv):
		print("""
			This command-line tool run concavity analysis tools from LSDTopoTools.
			Description of the algorithms in Mudd et al., 2018 -> https://www.earth-surf-dynam.net/6/505/2018/
			To use, run the script with relevant options, for example:
				lsdtt-concavity-down-to-top file=myraster.tif ---

			option available:
				file: name of the raster (file=name.tif)
				path: path to the file (default = current folder)
				todo
				help: if written, diplay this message. Documentation soon to be written.
			""")
		quit()

	try:
		default_param["X_source"] = float(default_param["X_source"])
		default_param["Y_source"] = float(default_param["Y_source"])
		default_param["min_elevation"] = float(default_param["min_elevation"])
		default_param["area_threshold"] = float(default_param["area_threshold"])
		default_param["flow_distance_step"] = float(default_param["flow_distance_step"])
		default_param["min_DA"] = float(default_param["min_DA"])
	except:
		print(default_param)
		print("I struggle to understand your input parameters: make sure I can convert them to number")
		raise SystemExit ("I need to quit now")

	mydem = LSDDEM(file_name = default_param["file"], path = default_param["path"], already_preprocessed = default_param["already_preprocessed"], remove_seas = True, sea_level = default_param["min_elevation"])

	if(default_param["already_preprocessed"] == False):
		print("I am preprocessing your raster with default options, see lsdtt-depressions for extended options.")
		print("Command line tools can load already processed rasters with the keyword already_preprocessed")
		mydem.PreProcessing()

	# flwo routines
	print("Processing flow routines with d8 algorithm")
	mydem.CommonFlowRoutines()

	print("Extracting the river")
	river = mydem.ExtractSingleRiverFromSource(default_param["X_source"],default_param["Y_source"])
	river = pd.DataFrame(river)
	print("Saving teh river to csv file: river_for_concavity.csv")
	river.to_csv("%sriver_for_concavity.csv"%(default_param["prefix"]), index = False)

	print("Initialising the concavity analysis")
	river.sort_values("elevation", inplace = True)
	river.reset_index(drop = True, inplace = True)

	# OUtputs
	global_results = {}
	outlet_info = {"X":[], "Y":[], "elevation":[], "flow_distance":[], "drainage_area":[]}

	# potential saving of basins
	XY_basins = {}

	flow_distance_this = river["flow_distance"].min()
	index_this = river["flow_distance"].idxmin()
	while(flow_distance_this < river["flow_distance"].max() and river["drainage_area"][index_this] > default_param["min_DA"]):
		print("processing  flow distance =", flow_distance_this)
		del mydem
		mydem = LSDDEM(file_name = "%sdem_for_concavity.tif"%(default_param["prefix"]), already_preprocessed = True, remove_seas =True, sea_level = river["elevation"][index_this]-5)
		mydem.CommonFlowRoutines()
		# mydem.save_array_to_raster_extent(mydem.cppdem.get_DA_raster(), name = "DA", save_directory = "./")

		mydem.ExtractRiverNetwork(area_threshold_min = default_param["area_threshold"])
		# print(river.iloc[index_this])
		mydem.DefineCatchment( method="from_XY", X_coords = [river["X"][index_this]], Y_coords = [river["Y"][index_this]], coord_search_radius_nodes = 0)
		mydem.GenerateChi()
		print(mydem.df_base_river["source_key"].unique())

		mydem.cppdem.calculate_movern_disorder(0.05, 0.05, 19, 1, default_param["area_threshold"])
		all_disorder = mydem.cppdem.get_best_fits_movern_per_BK()
		global_results[str(round(river["flow_distance"][index_this],2))] = np.array(all_disorder[0]) # [0] means basin 0, which is normal as I only have one
		outlet_info["X"].append(river["X"][index_this])
		outlet_info["Y"].append(river["Y"][index_this])
		outlet_info["elevation"].append(river["elevation"][index_this])
		outlet_info["flow_distance"].append(round(river["flow_distance"][index_this],2))
		outlet_info["drainage_area"].append(river["drainage_area"][index_this])
		
		# New index
		flow_distance_this += default_param["flow_distance_step"]
		index_this = river.index[river["flow_distance"]>=flow_distance_this].values[0]

		np.savez("%sdown_to_top_conc.npz"%(default_param["prefix"]), **global_results)
		pd.DataFrame(outlet_info).to_csv("%sdown_to_top_conc_basin_info.csv"%(default_param["prefix"]), index = False)

		if(default_param["save_XY_array"]):
			this_basin = mydem.cppdem.query_xy_for_each_basin()[0] # [0] because I want a single basin there
			print(this_basin)
			XY_basins[str(round(river["flow_distance"][index_this],2))] = this_basin
			np.savez(default_param["prefix"] + "down_to_top_basins_XY.npz", **XY_basins)





def concavity_from_csv():
	"""
	FUnction to autoamte the extraction of concavity using a csv file to read the basin name and outlets.
	This is useful for very specific concavity extraction.
	B.G
	"""
	pass
	# COPYPASTED FROM OTHER FILE, NEED TO ADAPT
	# default_param = AGPD.get_common_default_param()
	# default_param["already_preprocessed"] = False
	# default_param["X_source"] = None
	# default_param["Y_source"] = None
	# default_param["min_elevation"] = None
	# default_param["area_threshold"] = None
	# default_param["flow_distance_step"] = None
	# default_param["min_DA"] = None
	# default_param["prefix"] = ""
	# default_param = AGPD.ingest_param(default_param, sys.argv)


	# if(default_param["help"] or len(sys.argv)==1 or "help" in sys.argv):
	# 	print("""
	# 		This command-line tool run concavity analysis tools from LSDTopoTools.
	# 		Description of the algorithms in Mudd et al., 2018 -> https://www.earth-surf-dynam.net/6/505/2018/
	# 		To use, run the script with relevant options, for example:
	# 			lsdtt-concavity-down-to-top file=myraster.tif ---

	# 		option available:
	# 			file: name of the raster (file=name.tif)
	# 			path: path to the file (default = current folder)
	# 			todo
	# 			help: if written, diplay this message. Documentation soon to be written.
	# 		""")
	# 	quit()

	# try:
	# 	default_param["X_source"] = float(default_param["X_source"])
	# 	default_param["Y_source"] = float(default_param["Y_source"])
	# 	default_param["min_elevation"] = float(default_param["min_elevation"])
	# 	default_param["area_threshold"] = float(default_param["area_threshold"])
	# 	default_param["flow_distance_step"] = float(default_param["flow_distance_step"])
	# 	default_param["min_DA"] = float(default_param["min_DA"])
	# except:
	# 	print(default_param)
	# 	print("I struggle to understand your input parameters: make sure I can convert them to number")
	# 	raise SystemExit ("I need to quit now")	



