"""
concavity_automator comports multiple scripts automating concavity constraining method for landscape
"""
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
from scipy import spatial,stats
import numba as nb
import copy
from pathlib import Path
import pylab as pl
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def norm_by_row(A):
	"""
		Subfunction used to vectorised normalisation of disorder by max of row using apply_along_axis function
		B.G
	"""
	return A/A.max()

def norm_by_row_by_range(A):
	"""
		Subfunction used to vectorised normalisation of disorder by range of concavity using apply_along_axis function
		B.G
	"""
	return (A - A.min())/(A.max() - A.min())

def numfmt(x, pos):
	"""
	Plotting subfunction to automate tick formatting from metres to kilometres
	B.G
	"""
	s = '{:d}'.format(int(round(x / 1000.0)))
	return s

def get_best_bit_and_err_from_Dstar(thetas, medD, fstD, thdD):
	"""
		Takes ouput from concavity calculation to calculate the best-fit theta and its error
	"""
	# Calculating the index of minimum medium disorder to get the best-fit
	index_of_BF = np.argmin(medD)
	# Getting the Dstar value of the best-fit
	dstar_val = medD[index_of_BF]
	# Getting the acutal best-fit
	BF = thetas[index_of_BF]
	
	# Preformatting 2 arrays for calculating the error: I am just interested by the first half for the first error and the second for the second
	A = np.copy(fstD)
	A[index_of_BF+1:] = 9999
	B = np.copy(fstD)
	B[:index_of_BF] = 9999

	# calculating the error by extracting the closest theta with a Dstar close to the median best fit ones
	err = ( thetas[np.abs(A - dstar_val).argmin()] , thetas[np.abs(B - dstar_val).argmin()] )
	# REturning a tuple with [0] being the best fit and [1] another tuple f error
	return BF,err
	


def process_basin(ls, **kwargs):
	"""
		Main function processing the concavity. It looks a bit convoluted but it is required for clean multiprocessing.
		Takes at least one argument: ls, which is a list of arguments
			ls[0] -> the number of the basin (heavily used by automatic multiprocessing)
			ls[1] -> the X coordinate of the basin outlet
			ls[2] -> the Y coordinate of the basin outlet
			ls[3] -> area_threshold used for the analysis
			ls[4] -> prefix befor the number of the basin to read the file input
		Also takes option kwargs argument:
		ignore_numbering: jsut use the prefix as name for the DEM
		extension: if your extension is not .tif, you can give it here WITHOUT THE DOT
		overwrite_dem_name: used if you want to use thefunction from outside the automations: you need to provide the dem name WITH THE EXTENSION

	"""
	number = ls[0]
	X = ls[1]
	Y = ls[2]
	area_threshold = ls[3]
	prefix = ls[4]
	print("Processing basin ", number, " with proc ", current_process())
	if("ignore_numbering" not in kwargs):
		kwargs["ignore_numbering"] = False
	if("extension" not in kwargs):
		kwargs["extension"] = "tif"

	if("n_tribs_by_combo" not in kwargs):
		kwargs["n_tribs_by_combo"] = 4

	
	if(kwargs["ignore_numbering"] == True):
		name = prefix
	else:
		name = prefix + "%s"%(number)

	if(kwargs["precipitation_raster"] == ""):
		precipitation = False
	else:
		precipitation = True
	# I spent a significant amount of time preprocessing it, see SM
	n_rivers = 0
	dem_name ="%s.%s"%(name,kwargs["extension"])
	if("overwrite_dem_name" in kwargs):
		dem_name = kwargs["overwrite_dem_name"]

	MD = lsd.LSDDEM(file_name = dem_name, already_preprocessed = True)
	# Extracting basins
	if(precipitation):
		MD.CommonFlowRoutines( ingest_precipitation_raster = kwargs["precipitation_raster"], precipitation_raster_multiplier = 1, discharge = True)
	else:
		MD.CommonFlowRoutines()
	
	MD.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = area_threshold)
	print("River extracted")
	MD.DefineCatchment(  method="from_XY", X_coords = [X], Y_coords = [Y], coord_search_radius_nodes = 10 )#, X_coords = [X_coordinates_outlets[7]], Y_coords = [Y_coordinates_outlets[7]])
	print("CAtchment defined")
	MD.GenerateChi(theta = 0.4, A_0 = 1)
	print("River_network_generated")
	n_rivers = MD.df_base_river.source_key.unique().shape[0]
	print("You have", n_rivers, "rivers and",MD.df_base_river.shape[0],"river pixels")
	MD.df_base_river.to_feather("%s_rivers.feather"%(name))
	print("Starting the movern calculation")
	MD.cppdem.calculate_movern_disorder(0.05, 0.025, 38, 1, area_threshold, kwargs["n_tribs_by_combo"])
	print("DONE with movern, let's format the output")
	OVR_dis = MD.cppdem.get_disorder_dict()[0]
	OVR_tested = MD.cppdem.get_disorder_vec_of_tested_movern()
	pd.DataFrame({"overall_disorder":OVR_dis, "tested_movern":OVR_tested }).to_feather("%s_overall_test.feather"%(name))

	normalizer = MD.cppdem.get_n_pixels_by_combinations()[0]
	np.save("%s_disorder_normaliser.npy"%(name), normalizer)

	all_disorder = MD.cppdem.get_best_fits_movern_per_BK()
	np.save("%s_concavity_tot.npy"%(name), all_disorder[0])
	print("Getting results")
	results = np.array(MD.cppdem.get_all_disorder_values()[0])
	np.save("%s_disorder_tot.npy"%(name), results)
	XY = MD.cppdem.query_xy_for_each_basin()["0"]
	tdf = pd.DataFrame(XY)
	tdf.to_feather("%s_XY.feather"%(name))

	return 0

def theta_quick_constrain_single_basin(MD,X_coordinate_outlet = 0, Y_coordinate_outlet = 0, area_threshold = 1500):
	"""
		Main function processing the concavity. It looks a bit convoluted but it is required for clean multiprocessing.
		Takes at least one argument: ls, which is a list of arguments
			ls[0] -> the number of the basin (heavily used by automatic multiprocessing)
			ls[1] -> the X coordinate of the basin outlet
			ls[2] -> the Y coordinate of the basin outlet
			ls[3] -> area_threshold used for the analysis
			ls[4] -> prefix befor the number of the basin to read the file input
		Also takes option kwargs argument:
		ignore_numbering: jsut use the prefix as name for the DEM
		extension: if your extension is not .tif, you can give it here WITHOUT THE DOT
		overwrite_dem_name: used if you want to use thefunction from outside the automations: you need to provide the dem name WITH THE EXTENSION

	"""
	# number = ls[0]
	# X = ls[1]
	# Y = ls[2]
	# area_threshold = ls[3]
	# prefix = ls[4]
	# print("Processing basin ", number, " with proc ", current_process())
	# if("ignore_numbering" not in kwargs):
	# 	kwargs["ignore_numbering"] = False
	# if("extension" not in kwargs):
	# 	kwargs["extension"] = "tif"

	# if("n_tribs_by_combo" not in kwargs):
	# 	kwargs["n_tribs_by_combo"] = 4

	
	# if(kwargs["ignore_numbering"] == True):
	# 	name = prefix
	# else:
	# 	name = prefix + "%s"%(number)

	# if(kwargs["precipitation_raster"] == ""):
	# 	precipitation = False
	# else:
	# 	precipitation = True
	# I spent a significant amount of time preprocessing it, see SM
	n_rivers = 0
	# dem_name ="%s.%s"%(name,kwargs["extension"])
	# if("overwrite_dem_name" in kwargs):
	# 	dem_name = kwargs["overwrite_dem_name"]

	# MD = lsd.LSDDEM(file_name = dem_name, already_preprocessed = True)
	# # Extracting basins
	# if(precipitation):
	# 	MD.CommonFlowRoutines( ingest_precipitation_raster = kwargs["precipitation_raster"], precipitation_raster_multiplier = 1, discharge = True)
	# else:
	# 	MD.CommonFlowRoutines()
	# print("Experimental function (Gailleton et al., submitted), if it crashes restart from a clean LSDDEM object with only the flow routines processed.")
	MD.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = area_threshold)
	# print("River pre-extracted")
	MD.DefineCatchment(  method="from_XY", X_coords = X_coordinate_outlet, Y_coords = Y_coordinate_outlet, coord_search_radius_nodes = 10 )#, X_coords = [X_coordinates_outlets[7]], Y_coords = [Y_coordinates_outlets[7]])
	# print("CAtchment defined")
	MD.GenerateChi(theta = 0.4, A_0 = 1)
	# print("River_network_generated")
	n_rivers = MD.df_base_river.source_key.unique().shape[0]
	print("DEBUG::You have", n_rivers, "rivers and",MD.df_base_river.shape[0],"river pixels \n")
	# MD.df_base_river.to_feather("%s_rivers.feather"%(name))
	# print("Starting the movern calculation")
	MD.cppdem.calculate_movern_disorder(0.05, 0.025, 38, 1, area_threshold, 4)
	# print("DONE with movern, let's format the output")
	OVR_dis = MD.cppdem.get_disorder_dict()[0]
	OVR_tested = MD.cppdem.get_disorder_vec_of_tested_movern()
	# pd.DataFrame({"overall_disorder":OVR_dis, "tested_movern":OVR_tested }).to_feather("%s_overall_test.feather"%(name))

	normalizer = MD.cppdem.get_n_pixels_by_combinations()[0]
	# np.save("%s_disorder_normaliser.npy"%(name), normalizer)

	all_disorder = MD.cppdem.get_best_fits_movern_per_BK()
	# np.save("%s_concavity_tot.npy"%(name), all_disorder[0])
	# print("Getting results")
	results = np.array(MD.cppdem.get_all_disorder_values()[0])
	# np.save("%s_disorder_tot.npy"%(name), results)
	# XY = MD.cppdem.query_xy_for_each_basin()["0"]
	# tdf = pd.DataFrame(XY)
	# tdf.to_feather("%s_XY.feather"%(name))
	# print("\n\n")
	try:
		from IPython.display import display, Markdown, Latex
		todusplay = r"""
**Thanks for constraning** $\theta$ with the disorder algorithm from _Mudd et al., 2018_ and _Gailleton et al, submitted_.

Keep in mind that it is not straightforward and that the "best fit" we suggest is most of the time the "least worst" value maximising the collinearity in $\chi$ space. 

Especially in large, complex basin, several $\theta$ actually fit different areas and the best fit is just a try to make everyone happy where it is not necessarily possible.

$\theta$ constraining results:

median $\theta$ | $1^{st}$ Q | $3^{rd}$ Q
--- | --- | ---
%s | %s | %s
		"""%(round(np.nanmedian(all_disorder[0]),3), round(np.nanpercentile(all_disorder[0],25),3), round(np.nanpercentile(all_disorder[0],75),3))
		display(Markdown(todusplay))
	except:
		pass


	return all_disorder

def get_median_first_quartile_Dstar(ls):
	"""
		Function which post-process results from one analysis to return the median and first quartile curve of all best-fits
		param:
			ls: full prefix (= including basin number if needed)
		B.G
	"""
	print("Normalising D* for ", ls)

	name_to_load = ls
	# loading the file containng ALL the data
	all_data = np.load(name_to_load + "_disorder_tot.npy")
	if(all_data.shape[0]>1):
		# normalise by max each row
		all_data = np.apply_along_axis(norm_by_row,1,all_data)
		# Median by column
		ALLDmed = np.apply_along_axis(np.median,0,all_data)
		# Percentile by column
		ALLDfstQ = np.apply_along_axis(lambda z: np.percentile(z,25),0,all_data)
	else:
		return name_to_load

	return ALLDmed, ALLDfstQ, ls

def get_median_first_quartile_Dstar_r(ls):
	"""
		Function which post-process results from one analysis to return the median and first quartile curve of all best-fits
		param:
			ls: full prefix (= including basin number if needed)
		B.G
	"""
	print("Normalising D*_r for ", ls)

	name_to_load = ls
	# loading the file containng ALL the data
	all_data = np.load(name_to_load + "_disorder_tot.npy")
	if(all_data.shape[0]>1):
		# normalise by max each row
		all_data = np.apply_along_axis(norm_by_row_by_range,1,all_data)
		# Median by column
		ALLDmed = np.apply_along_axis(np.median,0,all_data)
		# Percentile by column
		ALLDfstQ = np.apply_along_axis(lambda z: np.percentile(z,25),0,all_data)
	else:
		return name_to_load

	return ALLDmed, ALLDfstQ, ls

def plot_single_theta(ls, **kwargs):
	"""
		For a multiple analysis on the same DEM this plot the global with each basins colored by D^*
		Need post-processing function to pre-analyse the ouputs.
		The layout of this function might seems a bit convoluted, but that's making multiprocessing easy, as they take time to plot
		param
	"""
	this_theta = ls[0]
	prefix = ls[1]

	# Loading the small summary df
	df = pd.read_csv(prefix +"summary_results.csv")
	# Loading the HillShade
	HS = lsd.raster_loader.load_raster(prefix + "HS.tif")

	# Formatting ticks
	import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
	yfmt = tkr.FuncFormatter(numfmt)
	xfmt = tkr.FuncFormatter(numfmt)

	print("plotting D* for theta", this_theta)

	# Getting the Figure and the ticks right
	fig,ax = plt.subplots()
	ax.yaxis.set_major_formatter(yfmt)
	ax.xaxis.set_major_formatter(xfmt)

	# Normalising the Hillshade and taking care of the no data
	HS["array"] = HS["array"]/HS["array"].max()
	HS["array"][HS["array"]<0] = np.nan

	# Plotting the hillshade
	ax.imshow(HS["array"], extent = HS["extent"], cmap = "gray", vmin= 0.2, vmax = 0.8)

	# Building the array of concavity
	A = np.zeros(HS["array"].shape)
	A[:,:] = np.nan
	# For each raster, I am reading rows and col corresponding to the main raster and potting it with the requested value
	for name in df["raster_name"]:
		row = np.load(name + "_row.npy")
		col =  np.load(name + "_col.npy")
		val = df["D*_%s"%this_theta][df["raster_name"] == name].values[0] #  A wee convoluted but it work and it is fast so...
		A[row,col] = val

	# PLOTTING THE D*
	ax.imshow(A, extent = HS["extent"], cmap= "gnuplot2", zorder = 2, alpha = 0.75, vmin = 0.1, vmax = 0.9)

	# You may want to change the extents of the plot

	if("xlim" in kwargs):
		ax.set_xlim(kwargs["xlim"])
	if("ylim" in kwargs):
		ax.set_ylim(kwargs["ylim"])

	ax.set_xlabel("Easting (km)")
	ax.set_ylabel("Northing (km)")

	# Saving the figure
	plt.tight_layout()#
	plt.savefig(prefix + "MAP_disorder_%s.png"%(this_theta), dpi = 500)
	plt.close(fig)

	print("plotting D*_r for theta", this_theta)

	# Getting the Figure and the ticks right
	fig,ax = plt.subplots()
	ax.yaxis.set_major_formatter(yfmt)
	ax.xaxis.set_major_formatter(xfmt)

	# Plotting the hillshade
	ax.imshow(HS["array"], extent = HS["extent"], cmap = "gray", vmin= 0.2, vmax = 0.8)

	# Building the array of concavity
	A = np.zeros(HS["array"].shape)
	A[:,:] = np.nan
	# For each raster, I am reading rows and col corresponding to the main raster and potting it with the requested value
	for name in df["raster_name"]:
		row = np.load(name + "_row.npy")
		col =  np.load(name + "_col.npy")
		val = df["D*_r_%s"%this_theta][df["raster_name"] == name].values[0] #  A wee convoluted but it work and it is fast so...
		A[row,col] = val

	# PLOTTING THE D*
	ax.imshow(A, extent = HS["extent"], cmap= "gnuplot2", zorder = 2, alpha = 0.75, vmin = 0.1, vmax = 0.9)

	# You may want to change the extents of the plot

	if("xlim" in kwargs):
		ax.set_xlim(kwargs["xlim"])
	if("ylim" in kwargs):
		ax.set_ylim(kwargs["ylim"])

	ax.set_xlabel("Easting (km)")
	ax.set_ylabel("Northing (km)")

	# Saving the figure
	plt.tight_layout()#
	plt.savefig(prefix + "MAP_disorder_by_range_%s.png"%(this_theta), dpi = 500)
	plt.close(fig)


	
	

def plot_min_D_star_map(ls, **kwargs):
	"""
		For a multiple analysis on the same DEM this plot the global with each basins colored by D^*
		Need post-processing function to pre-analyse the ouputs.
		The layout of this function might seems a bit convoluted, but that's making multiprocessing easy, as they take time to plot
		param
	"""
	this_theta = ls[0]
	prefix = ls[1]

	# Loading the small summary df
	df = pd.read_csv(prefix +"summary_results.csv")
	# Loading the HillShade
	HS = lsd.raster_loader.load_raster(prefix + "HS.tif")

	# Formatting ticks
	import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
	yfmt = tkr.FuncFormatter(numfmt)
	xfmt = tkr.FuncFormatter(numfmt)

	print("plotting D* for theta", this_theta)

	# Getting the Figure and the ticks right
	fig,ax = plt.subplots()
	ax.yaxis.set_major_formatter(yfmt)
	ax.xaxis.set_major_formatter(xfmt)

	# Normalising the Hillshade and taking care of the no data
	HS["array"] = HS["array"]/HS["array"].max()
	HS["array"][HS["array"]<0] = np.nan

	# Plotting the hillshade
	ax.imshow(HS["array"], extent = HS["extent"], cmap = "gray", vmin= 0.2, vmax = 0.8)

	# Building the array of concavity
	A = np.zeros(HS["array"].shape)
	A[:,:] = np.nan
	df_theta = pd.read_csv(prefix + "all_raster_names.csv")
	thetas = np.round(pd.read_feather(df["raster_name"].iloc[0] + "_overall_test.feather")["tested_movern"].values,decimals = 3)
 
	# For each raster, I am reading rows and col corresponding to the main raster and potting it with the requested value
	for name in df["raster_name"]:
		row = np.load(name + "_row.npy")
		col =  np.load(name + "_col.npy")
		val = 1e12
		for tval in thetas:
			valtest = df["D*_%s"%tval][df["raster_name"] == name].values[0] #  A wee convoluted but it work and it is fast so...
			if(valtest<val):
				val=valtest

		A[row,col] = val

	# PLOTTING THE D*
	ax.imshow(A, extent = HS["extent"], cmap= "gnuplot2", zorder = 2, alpha = 0.75, vmin = 0.1, vmax = 0.9)

	# You may want to change the extents of the plot

	if("xlim" in kwargs):
		ax.set_xlim(kwargs["xlim"])
	if("ylim" in kwargs):
		ax.set_ylim(kwargs["ylim"])

	ax.set_xlabel("Easting (km)")
	ax.set_ylabel("Northing (km)")

	# Saving the figure
	plt.tight_layout()#
	plt.savefig(prefix + "MAP_minimum_disorder_across_theta_%s.png"%(this_theta), dpi = 500)
	plt.close(fig)



def post_process_analysis_for_Dstar(prefix, n_proc = 1, base_raster_full_name = "SEC_PP.tif"):

	# Loading the list of raster
	df = pd.read_csv(prefix + "all_raster_names.csv")
	
	# Preparing the multiprocessing
	d_of_med = {}
	d_of_fst = {}
	d_of_med_r = {}
	d_of_fst_r = {}
	params = df["raster_name"].tolist()
	ras_to_ignore = {}
	ras_to_ignore_list = []

	for i in params:
		ras_to_ignore[i] = False

	# running the multiprocessing
	with Pool(n_proc) as p:
		fprocesses = []
		for i in params:
			fprocesses.append(p.apply_async(get_median_first_quartile_Dstar, args = (i,)))
		for gut in fprocesses:
			gut.wait()
		# getting the results in the right dictionaries
		for gut in fprocesses:
			# print(gut.get())
			if(isinstance(gut.get(),tuple)):
				d_of_med[gut.get()[2]] = gut.get()[0]
				d_of_fst[gut.get()[2]] = gut.get()[1]
			else:
				# print("IGNORING",gut.get() )
				ras_to_ignore[gut.get()] = True
				ras_to_ignore_list.append(gut.get())

	# running the multiprocessing
	with Pool(n_proc) as p:
		fprocesses = []
		for i in params:
			fprocesses.append(p.apply_async(get_median_first_quartile_Dstar_r, args = (i,)))
		for gut in fprocesses:
			gut.wait()
		# getting the results in the right dictionaries
		for gut in fprocesses:
			# print(gut.get())
			if(isinstance(gut.get(),tuple)):
				d_of_med_r[gut.get()[2]] = gut.get()[0]
				d_of_fst_r[gut.get()[2]] = gut.get()[1]
			else:
				# print("IGNORING",gut.get() )
				ras_to_ignore[gut.get()] = True
				ras_to_ignore_list.append(gut.get())



	# Getting the list of thetas tested
	thetas = np.round(pd.read_feather(params[0] + "_overall_test.feather")["tested_movern"].values,decimals = 3)


	df["best_fit"] = pd.Series(np.zeros(df.shape[0]), index = df.index)
	df["err_neg"] = pd.Series(np.zeros(df.shape[0]), index = df.index)
	df["err_pos"] = pd.Series(np.zeros(df.shape[0]), index = df.index)
	df["best_fit_norm_by_range"] = pd.Series(np.zeros(df.shape[0]), index = df.index)
	df["err_neg_norm_by_range"] = pd.Series(np.zeros(df.shape[0]), index = df.index)
	df["err_pos_norm_by_range"] = pd.Series(np.zeros(df.shape[0]), index = df.index)	

	# Preparing my dataframe to ingest
	for t in thetas:
		df["D*_%s"%t] = pd.Series(np.zeros(df.shape[0]), index = df.index)
		df["D*_r_%s"%t] = pd.Series(np.zeros(df.shape[0]), index = df.index)
	# Ingesting hte results
	for i in range(df.shape[0]):

		if(ras_to_ignore[df["raster_name"].iloc[i]]):
			continue
		
		BF,err = get_best_bit_and_err_from_Dstar(thetas, d_of_med[df["raster_name"].iloc[i]], d_of_fst[df["raster_name"].iloc[i]], 10)
		BF_r,err_r = get_best_bit_and_err_from_Dstar(thetas, d_of_med_r[df["raster_name"].iloc[i]], d_of_fst_r[df["raster_name"].iloc[i]], 10)
		df["best_fit"].iloc[i] = BF
		df["err_neg"].iloc[i] = err[0]
		df["err_pos"].iloc[i] = err[1]
		df["best_fit_norm_by_range"].iloc[i] = BF_r
		df["err_neg_norm_by_range"].iloc[i] =  err_r[0]
		df["err_pos_norm_by_range"].iloc[i] = err_r[1]

		for t in range(thetas.shape[0]):
			df["D*_%s"%thetas[t]].iloc[i] = d_of_med[df["raster_name"].iloc[i]][t]
			df["D*_r_%s"%thetas[t]].iloc[i] = d_of_med_r[df["raster_name"].iloc[i]][t]
	# Getting the hillshade
	mydem = lsd.LSDDEM(file_name = base_raster_full_name,already_preprocessed = True)
	HS = mydem.get_hillshade(altitude = 45, angle = 315, z_exageration = 1)
	mydem.save_array_to_raster_extent( HS, name = prefix + "HS", save_directory = "./")

	# will add X-Y to the sumarry dataframe
	df["X_median"] = pd.Series(np.zeros(df.shape[0]), index = df.index)
	df["X_firstQ"] = pd.Series(np.zeros(df.shape[0]), index = df.index)
	df["X_thirdtQ"] = pd.Series(np.zeros(df.shape[0]), index = df.index)
	df["Y_median"] = pd.Series(np.zeros(df.shape[0]), index = df.index)
	df["Y_firstQ"] = pd.Series(np.zeros(df.shape[0]), index = df.index)
	df["Y_thirdtQ"] = pd.Series(np.zeros(df.shape[0]), index = df.index)


	# I do not mutiprocess here: it would require load the mother raster for each process and would eat a lot of memory
	for i in params:
		if(ras_to_ignore[i]):
			continue
		XY = pd.read_feather(i + "_XY.feather")
		row,col = mydem.cppdem.query_rowcol_from_xy(XY["X"].values, XY["Y"].values)
		np.save(i + "_row.npy", row)
		np.save(i + "_col.npy", col)
		df["X_median"][df["raster_name"] == i] = XY["X"].median()
		df["X_firstQ"][df["raster_name"] == i] = XY["X"].quantile(0.25)
		df["X_thirdtQ"][df["raster_name"] == i] = XY["X"].quantile(0.75)
		df["Y_median"][df["raster_name"] == i] = XY["Y"].median()
		df["Y_firstQ"][df["raster_name"] == i] = XY["Y"].quantile(0.25)
		df["Y_thirdtQ"][df["raster_name"] == i] = XY["Y"].quantile(0.75)

	#Removing the unwanted
	df = df[~df["raster_name"].isin(ras_to_ignore_list)]
	
	# Saving the DataFrame
	df.to_csv(prefix +"summary_results.csv", index = False)
	print("Done with the post processing")

def plot_main_figures(prefix, **kwargs):
	# Loading the list of raster
	dfrast = pd.read_csv(prefix + "all_raster_names.csv")
	df = pd.read_csv(prefix +"summary_results.csv")
	# Creating the folder
	Path("./%s_figures"%(prefix)).mkdir(parents=True, exist_ok=True)
	print("Printing your histograms first")
	fig, ax = plt.subplots()
	ax.grid(ls = "--")
	ax.hist(df["best_fit"], bins = 19, histtype = "stepfilled", edgecolor = "k", facecolor = "orange", lw = 2)
	ax.set_xlabel(r"$\theta$")
	plt.tight_layout()
	plt.savefig("./%s_figures/%shistogram_all_fits.png"%(prefix, prefix), dpi = 500)
	plt.close(fig)

	print("Building the IQ CDF")
	IQR,bin_edge = np.histogram(df["err_pos"].values - df["err_neg"].values)
	fig, ax = plt.subplots()
	CSIQR = np.cumsum(IQR)
	CSIQR = CSIQR/np.nanmax(CSIQR)*100
	bin_edge = bin_edge[1:] - np.diff(bin_edge)
	ax.plot(bin_edge, CSIQR, lw = 2, color = "k", alpha = 1)

	# ax.axhspan(np.percentile(CSIQR,25),np.percentile(CSIQR,75), lw = 0, color = "r", alpha = 0.2)
	ax.fill_between(bin_edge,0,CSIQR, lw = 0, color = "k", alpha = 0.1)
	ax.set_xlabel(r"IQR $\theta$ best-fit")
	ax.set_ylabel(r"%")
	ax.grid(ls = "--", lw = 1)
	plt.savefig("./%s_figures/%sCDF_IQR.png"%(prefix, prefix), dpi = 500)
	plt.close(fig)


	print("plotting the map of best-fit")

	# Loading the small summary df
	df = pd.read_csv(prefix +"summary_results.csv")
	# Loading the HillShade
	HS = lsd.raster_loader.load_raster(prefix + "HS.tif")

	# Formatting ticks
	import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
	yfmt = tkr.FuncFormatter(numfmt)
	xfmt = tkr.FuncFormatter(numfmt)

	print("plotting best-fit")

	# Getting the Figure and the ticks right
	fig,ax = plt.subplots()
	ax.yaxis.set_major_formatter(yfmt)
	ax.xaxis.set_major_formatter(xfmt)

	# Normalising the Hillshade and taking care of the no data
	HS["array"] = HS["array"]/HS["array"].max()
	HS["array"][HS["array"]<0] = np.nan

	# Plotting the hillshade
	ax.imshow(HS["array"], extent = HS["extent"], cmap = "gray", vmin= 0.2, vmax = 0.8)

	# Building the array of concavity
	A = np.zeros(HS["array"].shape)
	A[:,:] = np.nan
	# For each raster, I am reading rows and col corresponding to the main raster and potting it with the requested value
	for name in df["raster_name"]:
		row = np.load(name + "_row.npy")
		col =  np.load(name + "_col.npy")
		val = df["best_fit"][df["raster_name"] == name].values[0] #  A wee convoluted but it work and it is fast so...
		A[row,col] = val

	# PLOTTING THE D*
	ax.imshow(A, extent = HS["extent"], cmap= "RdYlBu_r", zorder = 2, alpha = 0.75, vmin = 0.1, vmax = 0.9)

	# You may want to change the extents of the plot

	if("xlim" in kwargs):
		ax.set_xlim(kwargs["xlim"])
	if("ylim" in kwargs):
		ax.set_ylim(kwargs["ylim"])

	ax.set_xlabel("Easting (km)")
	ax.set_ylabel("Northing (km)")

	# Saving the figure
	plt.tight_layout()#
	plt.savefig("./%s_figures/"%(prefix) +prefix + "MAP_best_fit.png", dpi = 500)
	plt.close(fig)

	a = np.array([[0,1]])
	pl.figure(figsize=(9, 1.5))
	img = pl.imshow(a, cmap="RdYlBu_r")
	pl.gca().set_visible(False)
	cax = pl.axes([0.1, 0.2, 0.8, 0.6])
	pl.colorbar(orientation="horizontal", cax=cax)
	pl.title(r"$\theta$ best-fit")
	pl.savefig("./%s_figures/"%(prefix) +"colorbar_mapbest_fit.png")
	pl.close(fig)


	# Formatting ticks
	import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
	yfmt = tkr.FuncFormatter(numfmt)
	xfmt = tkr.FuncFormatter(numfmt)

	print("plotting min theta")

	# Getting the Figure and the ticks right
	fig,ax = plt.subplots()
	ax.yaxis.set_major_formatter(yfmt)
	ax.xaxis.set_major_formatter(xfmt)

	# Plotting the hillshade
	ax.imshow(HS["array"], extent = HS["extent"], cmap = "gray", vmin= 0.2, vmax = 0.8)

	# Building the array of concavity
	# Building the array of concavity
	A = np.zeros(HS["array"].shape)
	A[:,:] = np.nan
	df_theta = pd.read_csv(prefix + "all_raster_names.csv")
	thetas = np.round(pd.read_feather(df["raster_name"].iloc[0] + "_overall_test.feather")["tested_movern"].values,decimals = 3)
 
	# For each raster, I am reading rows and col corresponding to the main raster and potting it with the requested value
	for name in df["raster_name"]:
		row = np.load(name + "_row.npy")
		col =  np.load(name + "_col.npy")
		val = 1e12
		for tval in thetas:
			valtest = df["D*_%s"%tval][df["raster_name"] == name].values[0] #  A wee convoluted but it work and it is fast so...
			if(valtest<val):
				val=valtest

		A[row,col] = val

	# PLOTTING THE D*
	ax.imshow(A, extent = HS["extent"], cmap= "gnuplot2", zorder = 2, alpha = 0.75, vmin = 0.05, vmax = 0.65)

	# You may want to change the extents of the plot

	if("xlim" in kwargs):
		ax.set_xlim(kwargs["xlim"])
	if("ylim" in kwargs):
		ax.set_ylim(kwargs["ylim"])

	ax.set_xlabel("Easting (km)")
	ax.set_ylabel("Northing (km)")

	# Saving the figure
	plt.tight_layout()#
	plt.savefig("./%s_figures/"%(prefix) +prefix + "min_Dstar_for_each_basins.png", dpi = 500)
	plt.close(fig)

	a = np.array([[0,1]])
	pl.figure(figsize=(9, 1.5))
	img = pl.imshow(a, cmap="gnuplot2", vmin = 0.05, vmax = 0.65)
	pl.gca().set_visible(False)
	cax = pl.axes([0.1, 0.2, 0.8, 0.6])
	pl.colorbar(orientation="horizontal", cax=cax, label = r"Min. $D^{*}$")
	pl.title(r"Min. $D^{*}$")
	pl.savefig("./%s_figures/"%(prefix) +"colorbar_map_minDstar.png")
	pl.close(fig)



	# Formatting ticks
	import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
	yfmt = tkr.FuncFormatter(numfmt)
	xfmt = tkr.FuncFormatter(numfmt)

	print("plotting best-fit theta range yo")
	min_theta = 99999
	min_Dsum = 1e36
	for this_theta in thetas:
		this_sum = np.sum(df["D*_r_%s"%this_theta].values)
		if(this_sum < min_Dsum):
			min_theta = this_theta
			min_Dsum = this_sum

	this_theta = min_theta
	print("Which is ", this_theta)

	# Getting the Figure and the ticks right
	fig,ax = plt.subplots()
	ax.yaxis.set_major_formatter(yfmt)
	ax.xaxis.set_major_formatter(xfmt)

	# Plotting the hillshade
	ax.imshow(HS["array"], extent = HS["extent"], cmap = "gray", vmin= 0.2, vmax = 0.8)

	# Building the array of concavity
	A = np.zeros(HS["array"].shape)
	A[:,:] = np.nan
	# For each raster, I am reading rows and col corresponding to the main raster and potting it with the requested value
	for name in df["raster_name"]:
		row = np.load(name + "_row.npy")
		col =  np.load(name + "_col.npy")
		val = df["D*_r_%s"%this_theta][df["raster_name"] == name].values[0] #  A wee convoluted but it work and it is fast so...		A[row,col] = val
		A[row,col] = val

	# PLOTTING THE D*
	ax.imshow(A, extent = HS["extent"], cmap= "gnuplot2", zorder = 2, alpha = 0.75, vmin = 0.05, vmax = 0.65)

	# You may want to change the extents of the plot

	if("xlim" in kwargs):
		ax.set_xlim(kwargs["xlim"])
	if("ylim" in kwargs):
		ax.set_ylim(kwargs["ylim"])

	ax.set_xlabel("Easting (km)")
	ax.set_ylabel("Northing (km)")

	# Saving the figure
	plt.tight_layout()#
	plt.savefig("./%s_figures/"%(prefix) +prefix + "MAP_D_star_range_theta_%s.png" % this_theta, dpi = 500)
	plt.close(fig)


	a = np.array([[0,1]])
	pl.figure(figsize=(9, 1.5))
	img = pl.imshow(a, cmap="gnuplot2", vmin = 0.05, vmax = 0.65)
	pl.gca().set_visible(False)
	cax = pl.axes([0.1, 0.2, 0.8, 0.6])
	pl.colorbar(orientation="horizontal", cax=cax, label = r"$D^{*}_{r}$")
	pl.title(r"$D^{*}_{r}$")
	pl.savefig("./%s_figures/"%(prefix) +"colorbar_map_Dstar_range.png")
	pl.close(fig)





def plot_Dstar_maps_for_all_concavities(prefix, n_proc = 1):

	# Loading the list of raster
	df = pd.read_csv(prefix + "all_raster_names.csv")
	params = df["raster_name"].tolist()

	thetas = np.round(pd.read_feather(params[0] + "_overall_test.feather")["tested_movern"].values,decimals = 3) # running the multiprocessing
	
	params = []

	for t in thetas:
		params.append((t,prefix))

	# plot_single_theta(params[0])

	with Pool(n_proc) as p:
		fprocesses = []
		for i in params:
			fprocesses.append(p.apply_async(plot_single_theta, args = (i,)))
		for gut in fprocesses:
			gut.wait()
	plot_min_D_star_map(params[0])

def plot_basin(ls, **kwargs):
	number = ls[0]
	X = ls[1]
	Y = ls[2]
	area_threshold = ls[3]
	prefix = ls[4]
	nbins = None
	if("nbins" in kwargs):
		nbins = kwargs["nbins"]


	print("Plotting basin ", number, " with proc ", current_process())

	if("ignore_numbering" not in kwargs):
		kwargs["ignore_numbering"] = False

	if(kwargs["ignore_numbering"]):
		name = prefix
	else:
		name = prefix + "%s"%(number)


	# Alright, loading the previous datasets
	df_rivers = pd.read_feather("%s_rivers.feather"%(name))
	df_overall  = pd.read_feather("%s_overall_test.feather"%(name))
	all_concavity_best_fits = np.load("%s_concavity_tot.npy"%(name))
	all_disorders = np.load("%s_disorder_tot.npy"%(name))
	XY = pd.read_feather("%s_XY.feather"%(name))
	thetas = df_overall["tested_movern"].values

	if nbins is None:
		nbins = (thetas.shape[0], 50)

	res_dict = {}


	# Plotting the different figures for the basin
	# First, normalising the disorder
	AllDval = np.apply_along_axis(norm_by_row,1,all_disorders)
	AllDthet = np.tile(thetas,(all_disorders.shape[0],1))
	ALLDmed = np.apply_along_axis(np.median,0,AllDval)
	ALLDfstQ = np.apply_along_axis(lambda z: np.percentile(z,25),0,AllDval)
	ALLDthdQ = np.apply_along_axis(lambda z: np.percentile(z,75),0,AllDval)
	AllDval = AllDval.ravel()
	AllDthet = AllDthet.ravel()
	
	# Then, plotting it:
	###### First plotting the 
	fig,ax = plt.subplots()
	H,x,y = np.histogram2d(AllDthet,AllDval, bins = nbins, density = True)
	ax.hist2d(AllDthet,AllDval, bins = nbins, density = True, cmap = "magma", vmin = np.percentile(H,10), vmax = np.percentile(H,90))
	ax.plot(thetas,ALLDmed, lw = 1, ls = "-.", color = "#00F3FF")
	ax.plot(thetas,ALLDfstQ, lw = 1, ls = "--", color = "#00F3FF")
	ax.plot(thetas,ALLDthdQ, lw = 1, ls = "--", color = "#00F3FF")

	# Finding the suggested best-fit
	minimum_theta , flub = get_best_bit_and_err_from_Dstar(thetas, ALLDmed, ALLDfstQ, ALLDthdQ)
	err_neg,err_pos = flub

	res_dict["BF_normalised_disorder"] = minimum_theta
	res_dict["err_normalised_disorder"] = [err_neg,err_pos]

	print("Detected best-fit minimising normalised disorder is", minimum_theta, "tolerance between:", err_neg, "--",err_pos)

	ax.scatter(minimum_theta, np.min(ALLDmed), facecolor = "orange", edgecolor = "grey", s = 30, zorder = 3)
	ax.plot([err_neg,err_pos], [np.min(ALLDmed),np.min(ALLDmed)], color = "orange", lw = 2, zorder = 2)

	ax.set_xlabel(r"$\theta$")
	ax.set_ylabel(r"$D^{*}$")

	ax.set_xticks(np.arange(0.05,1,0.05))
	ax.set_yticks(np.arange(0.05,1,0.05))

	ax.grid(alpha = 0.3)
	ax.tick_params(labelsize = 8,)

	if("return_mode" not in kwargs):
		kwargs["return_mode"] = "save"

	if(kwargs["return_mode"].lower() == "save"):
		plt.savefig(name + "_D_star.png", dpi = 500)
		plt.close(fig)

	# Plotting the different figures for the basin
	# First, normalising the disorder by range
	AllDval = np.apply_along_axis(norm_by_row_by_range,1,all_disorders)
	AllDthet = np.tile(thetas,(all_disorders.shape[0],1))
	ALLDmed = np.apply_along_axis(np.median,0,AllDval)
	ALLDfstQ = np.apply_along_axis(lambda z: np.percentile(z,25),0,AllDval)
	ALLDthdQ = np.apply_along_axis(lambda z: np.percentile(z,75),0,AllDval)
	AllDval = AllDval.ravel()
	AllDthet = AllDthet.ravel()
	
	# Then, plotting it:
	###### First plotting the 
	fig,ax = plt.subplots()
	H,x,y = np.histogram2d(AllDthet,AllDval, bins = nbins, density = True)
	ax.hist2d(AllDthet,AllDval, bins = nbins, density = True, cmap = "magma", vmin = np.percentile(H,10), vmax = np.percentile(H,90))
	ax.plot(thetas,ALLDmed, lw = 1, ls = "-.", color = "#00F3FF")
	ax.plot(thetas,ALLDfstQ, lw = 1, ls = "--", color = "#00F3FF")
	ax.plot(thetas,ALLDthdQ, lw = 1, ls = "--", color = "#00F3FF")


	minimum_theta , flub = get_best_bit_and_err_from_Dstar(thetas, ALLDmed, ALLDfstQ, ALLDthdQ)
	err_neg,err_pos = flub
	# Finding the suggested best-fit


	res_dict["BF_normalised_disorder_range"] = minimum_theta
	res_dict["err_normalised_disorder_range"] = [err_neg,err_pos]

	print("Detected best-fit minimising normalised disorder is", minimum_theta, "tolerance between:", err_neg, "--",err_pos)

	ax.scatter(minimum_theta, np.min(ALLDmed), facecolor = "orange", edgecolor = "grey", s = 30, zorder = 3)
	ax.plot([err_neg,err_pos], [np.min(ALLDmed),np.min(ALLDmed)], color = "orange", lw = 2, zorder = 2)

	ax.set_xlabel(r"$\theta$")
	ax.set_ylabel(r"$D^{*}$")

	ax.set_xticks(np.arange(0.05,1,0.05))
	ax.set_yticks(np.arange(0.05,1,0.05))

	ax.grid(alpha = 0.3)
	ax.tick_params(labelsize = 8,)

	if("return_mode" not in kwargs):
		kwargs["return_mode"] = "save"

	if(kwargs["return_mode"].lower() == "save"):
		plt.savefig(name + "_D_star_norm_by_range.png", dpi = 500)
		plt.close(fig)

	#### Now plotting the rest
	
	df_overall = pd.read_feather("%s_overall_test.feather"%(name))
	res_dict["overall_best_fit"] = df_overall["tested_movern"][np.argmin(df_overall["overall_disorder"].values)]

	res_dict["median_all_lowest_values"] = np.median(all_concavity_best_fits)
	res_dict["IQ_all_lowest_values"] = [np.percentile(all_concavity_best_fits,25),np.percentile(all_concavity_best_fits,75)]
	u,c = np.unique(all_concavity_best_fits,return_counts = True)
	res_dict["max_combinations"] = u[np.argmax(c)]


	fig,ax = plt.subplots()
	# ax.hist(all_concavity_best_fits,bins = bins[0], edgecolor = "k", facecolor = "orange", zorder = 1, alpha = 0.3)
	ax.plot([0,0],res_dict["IQ_all_lowest_values"], color = "purple", lw = 2,zorder = 1)
	ax.scatter(0,res_dict["median_all_lowest_values"],edgecolor = "k", facecolor = "purple", s = 50, label = "Stats all values", zorder = 2)
	ax.scatter(1,res_dict["overall_best_fit"],edgecolor = "k", facecolor = "green", s = 50, label = "All values", zorder = 2)
	ax.scatter(2,res_dict["max_combinations"],edgecolor = "k", facecolor = "black", s = 50, label = "Max N tribs.", zorder = 2)
	ax.plot([3,3],res_dict["err_normalised_disorder"], color = "orange", lw = 2,zorder = 1)
	ax.scatter(3,res_dict["BF_normalised_disorder"],edgecolor = "k", facecolor = "orange", s = 50, label = r"$D^{*}$ norm. max.s", zorder = 2)
	ax.plot([4,4],res_dict["err_normalised_disorder_range"], color = "red", lw = 2,zorder = 1)
	ax.scatter(4,res_dict["BF_normalised_disorder_range"],edgecolor = "k", facecolor = "red", s = 50, label = r"$D^{*}$ norm. ranges", zorder = 2)
	ax.violinplot(all_concavity_best_fits,[5], showmeans  = False, showextrema =False,points = 100, bw_method= "silverman")
	# ax.legend()

	ax.set_xticks([0,1,2,3,4,5])
	ax.set_xticklabels(["Stats all values","All data best fit","Max N tribs.",r"$D^{*}$ norm. max.", r"$D^{*}$ norm. ranges", "data"])

	ax.set_yticks(np.round(np.arange(0.05,1,0.05), decimals = 2))

	ax.set_facecolor("grey")
	ax.grid(alpha = 0.5)

	ax.tick_params(axis = "x",labelrotation =45)
	ax.tick_params(axis = "both",labelsize = 8)

	# ax.hist(min_theta,bins = 38, edgecolor = "green", facecolor = "none", alpha = 0.6)
	ax.set_ylabel(r"$\theta$")
	plt.tight_layout()
	plt.savefig(name +"_all_best_fits_from_disorder_methods.png", dpi = 500)
	plt.close(fig)



	# elif(kwargs["return_mode"].lower() == "return"):
	# 	return fig,ax
	# elif(kwargs["return_mode"].lower() == "nothing"):
	# 	return 0

def get_all_concavity_in_range_of_DA_from_baselevel(dem_name, dem_path = "./",already_preprocessed = False , X_outlet = 0, Y_outlet = 0, 
	min_DA = 1e7, max_DA = 2e8, area_threshold = 2000,area_threshold_main_basin = 25000 , n_proc = 4, prefix = "", n_tribs_by_combo = 4):

	print("First, elt me extract all the basins")
	# First, I need to extract the basin into separated rasters
	X = X_outlet
	Y = Y_outlet

	# Loading the dem
	mydem = lsd.LSDDEM(file_name = dem_name, path = dem_path, already_preprocessed = already_preprocessed)

	# Preprocessing
	if(already_preprocessed == False):
		mydem.PreProcessing()

	# Getting DA and otehr stuffs
	mydem.CommonFlowRoutines()
	A = mydem.cppdem.get_DA_raster()
	mydem.save_array_to_raster_extent( A, name = prefix + "drainage_area")
	# This define the river network, it is required to actually calculate other metrics
	mydem.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = 800)
	print("DEBUG::RIVEREXTRACTED")
	
	# Extracting all the basins	
	coord_bas = mydem.cppdem.calculate_outlets_min_max_draining_to_baselevel(X, Y, min_DA, max_DA,500)
	print("DEBUG::BASINEXTRACTED")
	print(coord_bas)

	# Getting the rasters
	rasts = mydem.cppdem.get_individual_basin_raster()
	coord_bas["ID"] = list(range(len(rasts[0])))

	for i in range(len(rasts[0])):
		lsd.raster_loader.save_raster(rasts[0][i],rasts[1][i]["x_min"],rasts[1][i]["x_max"],rasts[1][i]["y_max"],rasts[1][i]["y_min"],rasts[1][i]["res"],mydem.crs, prefix + "%s.tif"%(i), fmt = 'GTIFF')

	df = pd.DataFrame(coord_bas)
	df.to_csv(prefix + "basin_outlets.csv", index = False)


	# Freeing memory
	del mydem
	del rasts


	print("Done with basin extraction, I am now multiprocessing the concavity analysis, this can take a while if you have a high number of river!")
	th = np.full(df["ID"].shape[0],area_threshold)
	tprefix = np.full(df["ID"].shape[0],prefix)
	N = list(zip(df["ID"].values,df["X"].values,df["Y"].values, th, tprefix)) # for single analysis
	these_kwarges = []
	for i in N:
		these_kwarges.append({"n_tribs_by_combo":n_tribs_by_combo})
	# print(N)
	p = Pool(n_proc)
	# results = p.map_async(process_basin, N)
	# results.wait()
	# p.close()
	# p.join()
	with Pool(n_proc) as p:
		fprocesses = []
		for i in range(len(N)):
			fprocesses.append(p.apply_async(process_basin, args = (N[i],),kwds = these_kwarges[i]))
		for gut in fprocesses:
			gut.wait()
		# A = p.get()

	print("Done with all the sub basins, now I will process the main basin")

def process_multiple_basins(dem_name, dem_path = "./",already_preprocessed = False , prefix = "", X_outlets = [0], Y_outlets = [0], n_proc = 1, area_threshold = [5000], 
	area_thershold_basin_extraction = 500, n_tribs_by_combo = 5,use_precipitation_raster = False, name_precipitation_raster = "prec.tif"):



	# IDs = np.array(IDs)
	X_outlets = np.array(X_outlets)
	Y_outlets = np.array(Y_outlets)
	area_threshold = np.array(area_threshold)


	mydem = lsd.LSDDEM(file_name = dem_name, path = dem_path, already_preprocessed = already_preprocessed)

	# Preprocessing
	if(already_preprocessed == False):
		mydem.PreProcessing()

	# Getting DA and otehr stuffs
	mydem.CommonFlowRoutines()
	A = mydem.cppdem.get_DA_raster()
	mydem.save_array_to_raster_extent( A, name = prefix + "drainage_area")
	# This define the river network, it is required to actually calculate other metrics
	mydem.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = area_thershold_basin_extraction)
	mydem.DefineCatchment( method="from_XY", X_coords = X_outlets, Y_coords = Y_outlets, coord_search_radius_nodes = 0)

	# Getting the rasters
	rasts = mydem.cppdem.get_individual_basin_raster()
	# IDs = np.array(IDs)
	IDs = np.array(range(len(rasts[0])))
	out_names = {"raster_name": []}


	for i in range(len(rasts[0])):
		lsd.raster_loader.save_raster(rasts[0][i],rasts[1][i]["x_min"],rasts[1][i]["x_max"],rasts[1][i]["y_max"],rasts[1][i]["y_min"],rasts[1][i]["res"],mydem.crs, prefix + "%s.tif"%(i), fmt = 'GTIFF')
		out_names["raster_name"].append(prefix + "%s"%(i))

	pd.DataFrame(out_names).to_csv(prefix + "all_raster_names.csv", index = False)

	del mydem
	del rasts

	th = np.full(IDs.shape[0],area_threshold)
	tprefix = np.full(IDs.shape[0],prefix)
	N = list(zip(IDs,X_outlets,Y_outlets, th, tprefix)) # for single analysis
	these_kwarges = []
	for i in N:
		dico ={}
		dico["n_tribs_by_combo"] = n_tribs_by_combo
		if(use_precipitation_raster):
			dico["precipitation_raster"]=dem_path+name_precipitation_raster
		else:
			dico["precipitation_raster"]=""
		these_kwarges.append(dico)


	# print(N)
	p = Pool(n_proc)
	# results = p.map_async(process_basin, N)
	# results.wait()
	# p.close()
	# p.join()
	for i in range(len(N)):
		process_basin(N[i],**these_kwarges[i])
	# with Pool(n_proc) as p:
	# 	fprocesses = []
	# 	for i in range(len(N)):
	# 		fprocesses.append(p.apply_async(process_basin, args = (N[i],),kwds = these_kwarges[i]))
	# 	for gut in fprocesses:
	# 		gut.wait()


def plot_one_thetha(ls):
	import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
	yfmt = tkr.FuncFormatter(numfmt)
	xfmt = tkr.FuncFormatter(numfmt)
	that_theta = ls[0]

	print("plotting D* for theta", that_theta)
	fig,ax = plt.subplots()
	ax.yaxis.set_major_formatter(yfmt)
	ax.xaxis.set_major_formatter(xfmt)
	ax.imshow(ls[1],extent = ls[2], vmin = 0, vmax = 1, cmap = "gray", zorder = 1)
	cb = ax.imshow(ls[3],extent = ls[2], vmin = 0.1, vmax = 0.9, cmap = "magma_r", zorder = 2, alpha = 0.8)
	plt.colorbar(cb, orientation = "horizontal", label = r"$D^{*}$ for $\theta=%s$"%(round(that_theta,3)))
	ax.set_xlabel("Easting (km)")
	ax.set_xlabel("Northing (km)")

	plt.savefig(ls[4], dpi = 500)
	plt.close(fig)
	print("Done plotting D* for theta", that_theta)

def plot_multiple_basins(dem_name, dem_path = "./",already_preprocessed = False , prefix = "", X_outlets = [0], Y_outlets = [0],
 n_proc = 1, area_threshold = [5000], area_thershold_basin_extraction = 500, plot_Dstar = False):
	"""
	This function plot the data for a list of basins
	"""


	# reloading the df needed ro process the output
	df = pd.read_csv(prefix + "summary_results.csv")

	# Outputs stored in lists
	df_rivers = []
	df_overall = []
	all_concavity_best_fits = []
	all_disorders = []
	XY = []
	thetas = []

	for i in range(df["raster_name"].shape[0]):
		name = df["raster_name"].iloc[i]
		# Alright, loading the previous datasets
		df_rivers.append(pd.read_feather("%s_rivers.feather"%(name)))
		df_overall.append(pd.read_feather("%s_overall_test.feather"%(name)))
		all_concavity_best_fits.append(np.load("%s_concavity_tot.npy"%(name)))
		all_disorders.append(np.load("%s_disorder_tot.npy"%(name)))
		XY.append(pd.read_feather("%s_XY.feather"%(name)))
	thetas = (df_overall[-1]["tested_movern"].values)

	size_of_stuff = len(df_rivers)

	# First, a condensed of all informations
	easting = []
	northing = []
	easting_err = []
	northing_err = []
	best_fits_Dstar = []
	err_Dstar = []
	min_Dstar = []
	for i in range(size_of_stuff):
		AllDval = np.apply_along_axis(norm_by_row,1,all_disorders[i])
		AllDthet = np.tile(thetas,(all_disorders[i].shape[0],1))
		ALLDmed = np.apply_along_axis(np.median,0,AllDval)
		ALLDfstQ = np.apply_along_axis(lambda z: np.percentile(z,25),0,AllDval)
		ALLDthdQ = np.apply_along_axis(lambda z: np.percentile(z,75),0,AllDval)
		AllDval = AllDval.ravel()
		AllDthet = AllDthet.ravel()
		# Finding the suggested best-fit
		minimum_theta , flub = get_best_bit_and_err_from_Dstar(thetas, ALLDmed, ALLDfstQ, ALLDthdQ)
		err_neg,err_pos = flub

		easting.append(np.median(XY[i]["X"]))
		northing.append(np.median(XY[i]["Y"]))
		easting_err.append([np.percentile(XY[i]["X"],25),np.percentile(XY[i]["X"],75)])
		northing_err.append([np.percentile(XY[i]["Y"],25),np.percentile(XY[i]["Y"],75)])
		best_fits_Dstar.append(minimum_theta)
		min_Dstar.append(np.min(ALLDmed))
		err_Dstar.append([err_neg,err_pos])
	easting = np.array(easting)
	northing = np.array(northing)
	easting_err = np.array(easting_err)
	northing_err = np.array(northing_err)
	best_fits_Dstar = np.array(best_fits_Dstar)
	err_Dstar = np.array(err_Dstar)
	min_Dstar = np.array(min_Dstar)

	fig,ax = plt.subplots()

	ax.scatter(easting, best_fits_Dstar, edgecolor = "k", facecolor = "orange", s = 50, zorder = 5)
	for i in range(size_of_stuff):
		ax.plot([easting[i], easting[i]], err_Dstar[i], color = "orange", lw = 2, zorder = 1, alpha = 0.7)
		ax.plot(easting_err[i], [best_fits_Dstar[i],best_fits_Dstar[i]], color = "orange", lw = 2, zorder = 1, alpha = 0.7)

	xticks = np.arange(easting.min(), easting.max()+1, (easting.max() - easting.min())/5)

	ax.set_xticks(xticks)
	xticks = xticks / 1000
	ax.set_xticklabels(np.round(xticks).astype(str))

	ax.set_xlabel(r"Easting (km)")
	ax.set_ylabel(r"$\theta$")

	ax.set_facecolor("grey")
	ax.grid(zorder = 1, ls = "--", alpha = 0.7)

	plt.savefig(prefix + "_best_fit_by_easting", dpi=500)
	plt.close()

	fig,ax = plt.subplots()

	ax.scatter(northing, best_fits_Dstar, edgecolor = "k", facecolor = "orange", s = 50, zorder = 5)
	for i in range(size_of_stuff):
		ax.plot([northing[i], northing[i]], err_Dstar[i], color = "orange", lw = 2, zorder = 1, alpha = 0.7)
		ax.plot(northing_err[i], [best_fits_Dstar[i],best_fits_Dstar[i]], color = "orange", lw = 2, zorder = 1, alpha = 0.7)
	
	xticks = np.arange(northing.min(), northing.max()+1, (northing.max() - northing.min())/5)

	ax.set_xticks(xticks)
	xticks = xticks / 1000
	ax.set_xticklabels(np.round(xticks).astype(str))

	ax.set_xlabel(r"Northing (km)")
	ax.set_ylabel(r"$\theta$")

	ax.set_facecolor("grey")
	ax.grid(zorder = 1, ls = "--", alpha = 0.7)

	plt.savefig(prefix + "_best_fit_by_northing", dpi=500)
	plt.close()

	fig, ax = plt.subplots()
	ax.grid()
	ax.hist(best_fits_Dstar, bins = 18, histtype = "stepfilled", edgecolor = "k", facecolor = "orange", lw = 1.5)
	ax.set_xlabel(r"$\theta$ best-fits")
	plt.savefig(prefix + "_best_fit_histogram.png", dpi = 500)
	plt.close(fig)

	fig, ax = plt.subplots()
	ax.grid()
	ax.hist(min_Dstar, bins = 18, histtype = "stepfilled", edgecolor = "k", facecolor = "red", lw = 1.5)
	ax.set_xlabel(r"minimum $D^{*}$")
	plt.savefig(prefix + "_minimum_D_star_histogram.png", dpi = 500)
	plt.close(fig)


	fig, ax = plt.subplots()
	ax.grid()
	ax.hist(err_Dstar[:,1]-err_Dstar[:,0], bins = 18, histtype = "stepfilled", edgecolor = "k", facecolor = "limegreen", lw = 1.5)
	ax.set_xlabel(r"Inter-Quartile uncertainty")
	plt.savefig(prefix + "_IQ_uncert_histogram.png", dpi = 500)
	plt.close(fig)


	import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
	yfmt = tkr.FuncFormatter(numfmt)
	xfmt = tkr.FuncFormatter(numfmt)

	# Now dealing with the map
	mydem = lsd.LSDDEM(file_name = dem_name, path = dem_path, already_preprocessed = True)
	HS = mydem.get_hillshade()
	HS = HS/HS.max()
	array_of_concavities = np.copy(HS)
	array_of_concavities[:,:] = np.nan
	row_all = []
	col_all = []
	for i in range(size_of_stuff):
		Xs = XY[i]["X"].values
		Ys = XY[i]["Y"].values
		row,col = mydem.cppdem.query_rowcol_from_xy(Xs, Ys)
		array_of_concavities[row,col] = best_fits_Dstar[i]
		row_all.append(row)
		col_all.append(col)

	fig,ax = plt.subplots()
	ax.yaxis.set_major_formatter(yfmt)
	ax.xaxis.set_major_formatter(xfmt)

	HS[HS < 0] = np.nan
	array_of_concavities[array_of_concavities < 0] = np.nan

	ax.imshow(HS,extent = mydem.extent, vmin = 0, vmax = 1, cmap = "gray", zorder = 1)
	cb = ax.imshow(array_of_concavities,extent = mydem.extent, vmin = 0.1, vmax = 0.9, cmap = "RdBu_r", zorder = 2, alpha = 0.8)
	plt.colorbar(cb, orientation = "horizontal", label = r"$\theta$ best-fit")
	ax.set_xlabel("Easting (km)")
	ax.set_xlabel("Northing (km)")

	plt.savefig(prefix+"_map_all_concavities.png", dpi = 500)
	plt.close(fig)

	############################################################################################################################################################
	########################################################Plotting for range now##############################################################################
	############################################################################################################################################################

	# reloading the df needed ro process the output
	df = pd.read_csv(prefix + "summary_results.csv")

	# Outputs stored in lists
	df_rivers = []
	df_overall = []
	all_concavity_best_fits = []
	all_disorders = []
	XY = []
	thetas = []

	for i in range(df["raster_name"].shape[0]):
		name = df["raster_name"].iloc[i]
		# Alright, loading the previous datasets
		df_rivers.append(pd.read_feather("%s_rivers.feather"%(name)))
		df_overall.append(pd.read_feather("%s_overall_test.feather"%(name)))
		all_concavity_best_fits.append(np.load("%s_concavity_tot.npy"%(name)))
		all_disorders.append(np.load("%s_disorder_tot.npy"%(name)))
		XY.append(pd.read_feather("%s_XY.feather"%(name)))
	thetas = (df_overall[-1]["tested_movern"].values)

	size_of_stuff = len(df_rivers)

	# First, a condensed of all informations
	easting = []
	northing = []
	easting_err = []
	northing_err = []
	best_fits_Dstar = []
	err_Dstar = []
	for i in range(size_of_stuff):
		AllDval = np.apply_along_axis(norm_by_row_by_range,1,all_disorders[i])
		AllDthet = np.tile(thetas,(all_disorders[i].shape[0],1))
		ALLDmed = np.apply_along_axis(np.median,0,AllDval)
		ALLDfstQ = np.apply_along_axis(lambda z: np.percentile(z,25),0,AllDval)
		ALLDthdQ = np.apply_along_axis(lambda z: np.percentile(z,75),0,AllDval)
		AllDval = AllDval.ravel()
		AllDthet = AllDthet.ravel()
		# Finding the suggested best-fit
		minimum_theta , flub = get_best_bit_and_err_from_Dstar(thetas, ALLDmed, ALLDfstQ, ALLDthdQ)
		err_neg,err_pos = flub

		easting.append(np.median(XY[i]["X"]))
		northing.append(np.median(XY[i]["Y"]))
		easting_err.append([np.percentile(XY[i]["X"],25),np.percentile(XY[i]["X"],75)])
		northing_err.append([np.percentile(XY[i]["Y"],25),np.percentile(XY[i]["Y"],75)])
		best_fits_Dstar.append(minimum_theta)
		err_Dstar.append([err_neg,err_pos])
	easting = np.array(easting)
	northing = np.array(northing)
	easting_err = np.array(easting_err)
	northing_err = np.array(northing_err)
	best_fits_Dstar = np.array(best_fits_Dstar)
	err_Dstar = np.array(err_Dstar)

	fig,ax = plt.subplots()

	ax.scatter(easting, best_fits_Dstar, edgecolor = "k", facecolor = "orange", s = 50, zorder = 5)
	for i in range(size_of_stuff):
		ax.plot([easting[i], easting[i]], err_Dstar[i], color = "orange", lw = 2, zorder = 1, alpha = 0.7)
		ax.plot(easting_err[i], [best_fits_Dstar[i],best_fits_Dstar[i]], color = "orange", lw = 2, zorder = 1, alpha = 0.7)

	xticks = np.arange(easting.min(), easting.max()+1, (easting.max() - easting.min())/5)

	ax.set_xticks(xticks)
	xticks = xticks / 1000
	ax.set_xticklabels(np.round(xticks).astype(str))

	ax.set_xlabel(r"Easting (km)")
	ax.set_ylabel(r"$\theta$")

	ax.set_facecolor("grey")
	ax.grid(zorder = 1, ls = "--", alpha = 0.7)

	plt.savefig(prefix + "_best_fit_by_range_by_easting.png", dpi=500)
	plt.close()

	fig,ax = plt.subplots()

	ax.scatter(northing, best_fits_Dstar, edgecolor = "k", facecolor = "orange", s = 50, zorder = 5)
	for i in range(size_of_stuff):
		ax.plot([northing[i], northing[i]], err_Dstar[i], color = "orange", lw = 2, zorder = 1, alpha = 0.7)
		ax.plot(northing_err[i], [best_fits_Dstar[i],best_fits_Dstar[i]], color = "orange", lw = 2, zorder = 1, alpha = 0.7)
	
	xticks = np.arange(northing.min(), northing.max()+1, (northing.max() - northing.min())/5)

	ax.set_xticks(xticks)
	xticks = xticks / 1000
	ax.set_xticklabels(np.round(xticks).astype(str))

	ax.set_xlabel(r"Northing (km)")
	ax.set_ylabel(r"$\theta$")

	ax.set_facecolor("grey")
	ax.grid(zorder = 1, ls = "--", alpha = 0.7)

	plt.savefig(prefix + "_best_fit_by_range_by_northing", dpi=500)
	plt.close()


	import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
	yfmt = tkr.FuncFormatter(numfmt)
	xfmt = tkr.FuncFormatter(numfmt)

	# Now dealing with the map
	mydem = lsd.LSDDEM(file_name = dem_name, path = dem_path, already_preprocessed = True)
	HS = mydem.get_hillshade()
	HS = HS/HS.max()
	array_of_concavities = np.copy(HS)
	array_of_concavities[:,:] = np.nan
	row_all = []
	col_all = []
	for i in range(size_of_stuff):
		Xs = XY[i]["X"].values
		Ys = XY[i]["Y"].values
		row,col = mydem.cppdem.query_rowcol_from_xy(Xs, Ys)
		array_of_concavities[row,col] = best_fits_Dstar[i]
		row_all.append(row)
		col_all.append(col)

	fig,ax = plt.subplots()
	ax.yaxis.set_major_formatter(yfmt)
	ax.xaxis.set_major_formatter(xfmt)

	HS[HS < 0] = np.nan
	array_of_concavities[array_of_concavities < 0] = np.nan

	ax.imshow(HS,extent = mydem.extent, vmin = 0, vmax = 1, cmap = "gray", zorder = 1)
	cb = ax.imshow(array_of_concavities,extent = mydem.extent, vmin = 0.1, vmax = 0.9, cmap = "RdBu_r", zorder = 2, alpha = 0.8)
	plt.colorbar(cb, orientation = "horizontal", label = r"$\theta$ best-fit")
	ax.set_xlabel("Easting (km)")
	ax.set_xlabel("Northing (km)")
	plt.savefig(prefix+"_map_all_concavities_norm_by_range.png", dpi = 500)
	plt.close(fig)

	fig, ax = plt.subplots()
	ax.grid()
	ax.hist(best_fits_Dstar, bins = 18, histtype = "stepfilled", edgecolor = "k", facecolor = "orange", lw = 1.5)
	ax.set_xlabel(r"$\theta$ best-fits")
	plt.savefig(prefix + "_best_fit_by_range_histogram.png", dpi = 500)
	plt.close(fig)

	fig, ax = plt.subplots()
	ax.grid()
	ax.hist(err_Dstar[:,1]-err_Dstar[:,0], bins = 18, histtype = "stepfilled", edgecolor = "k", facecolor = "limegreen", lw = 1.5)
	ax.set_xlabel(r"Inter-Quartile uncertainty")
	plt.savefig(prefix + "_IQ_uncert_by_range_histogram.png", dpi = 500)
	plt.close(fig)

	############################################################################################################################################################
	################################################################## Following is deprecated #################################################################
	############################################################################################################################################################
	# Stop the function here if no plotting of the D* thingy (takes time)
	if(plot_Dstar == False):
		return 0 

	these_array = []
	for this_theta in range(thetas.shape[0]):
		array_of_concavities[:,:] = np.nan
		these_array.append(np.copy(array_of_concavities))

	# And finally all the figure by D*
	for i in range(size_of_stuff):	
		print("Calculating D* for basin", i)
		AllDval = np.apply_along_axis(norm_by_row,1,all_disorders[i])
		AllDthet = np.tile(thetas,(all_disorders[i].shape[0],1))
		ALLDmed = np.apply_along_axis(np.median,0,AllDval)
		Xs = XY[i]["X"].values
		Ys = XY[i]["Y"].values
		row,col = row_all[i],col_all[i]

		for this_theta in range(thetas.shape[0]):
			that_theta = thetas[this_theta]			
			these_array[this_theta][row,col] = ALLDmed[this_theta]
			
	

	params = []
	for this_theta in range(thetas.shape[0]):
		params.append([thetas[this_theta], HS,mydem.extent, these_array[this_theta], prefix+"_map_Dstar_%s.png"%(round(thetas[this_theta],3))])
		# print("plotting D* for theta", thetas[this_theta])
		# that_theta = thetas[this_theta]
		# fig,ax = plt.subplots()
		# ax.yaxis.set_major_formatter(yfmt)
		# ax.xaxis.set_major_formatter(xfmt)
		# ax.imshow(HS,extent = mydem.extent, vmin = 0, vmax = 1, cmap = "gray", zorder = 1)
		# cb = ax.imshow(these_array[this_theta],extent = mydem.extent, vmin = 0.1, vmax = 0.9, cmap = "magma_r", zorder = 2, alpha = 0.8)
		# plt.colorbar(cb, orientation = "horizontal", label = r"$D^{*}$ for $\theta=%s$"%(round(that_theta,3)))
		# ax.set_xlabel("Easting (km)")
		# ax.set_xlabel("Northing (km)")

		# plt.savefig(prefix+"_map_Dstar_%s.png"%(round(that_theta,3)), dpi = 500)
		# plt.close(fig)

	for i in params:
		plot_one_thetha(i)

	# with Pool(n_proc) as p:
	# 	print("Flub?")
	# 	fprocesses = []
	# 	for i in params:
	# 		print("Flub?", i)
	# 		fprocesses.append(p.apply_async(plot_one_thetha, args = (i,)))

		# for gut in fprocesses:
		# 	gut.wait()





	########################################################################################################################################################################################################################################################################################################################
	########################################################################################################################################################################################################################################################################################################################









def process_main_basin(dem_name, dem_path = "./", already_preprocessed = False , X_outlet = 0, Y_outlet = 0, area_threshold = 5000, area_threshold_to_keep_river = 100000, prefix = "", n_tribs_by_combo = 4):
	"""
	DEPRECATED
	"""	

	# while(n_rivers <20 and area_threshold > 200 ):
	X = X_outlet
	Y = Y_outlet

	# Loading the dem
	mydem = lsd.LSDDEM(file_name = dem_name, path = dem_path, already_preprocessed = already_preprocessed)
	if(already_preprocessed == False):
		mydem.PreProcessing()
	# Extracting basins
	mydem.CommonFlowRoutines()
	mydem.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = area_threshold_to_keep_river)
	mydem.DefineCatchment(  method="from_XY", X_coords = [X], Y_coords = [Y], coord_search_radius_nodes = 500 )#, X_coords = [X_coordinates_outlets[7]], Y_coords = [Y_coordinates_outlets[7]])
	mydem.GenerateChi(theta = 0.4, A_0 = 1)

	# sources = []
	# # Calculating rivers
	# rdf = mydem.df_base_river
	# for SK in rdf["source_key"].unique():
	# 	checker = float(rdf["drainage_area"][rdf["source_key"] == SK].max())
	# 	if(checker >= area_threshold_to_keep_river):
	# 		this_node_index = int(rdf["nodeID"][rdf["drainage_area"] == rdf["drainage_area"][rdf["source_key"] == SK].min()].iloc[0])
	# 		sources.append(this_node_index)

	# del mydem
	# # Loading the dem
	# mydem = lsd.LSDDEM(file_name = dem_name, path = dem_path, already_preprocessed = already_preprocessed)
	# if(already_preprocessed == False):
	# 	mydem.PreProcessing()
	# # Extracting basins
	# mydem.CommonFlowRoutines()
	# mydem.ExtractRiverNetwork( method = "source_nodes", source_nodes = np.array(sources))
	
	# mydem.DefineCatchment(  method="from_XY", X_coords = [X], Y_coords = [Y], coord_search_radius_nodes = 500 )#, X_coords = [X_coordinates_outlets[7]], Y_coords = [Y_coordinates_outlets[7]])
	# mydem.GenerateChi(theta = 0.4, A_0 = 1)

	name = "main_basin"

	mydem.df_base_river.to_csv(prefix +"main_basin_rivers.csv", index = False)
	n_rivers = mydem.df_base_river.source_key.unique().shape[0]
	mydem.cppdem.calculate_movern_disorder(0.05, 0.05, 19, 1, area_threshold, n_tribs_by_combo)
	OVR_dis = mydem.cppdem.get_disorder_dict()[0]
	OVR_tested = mydem.cppdem.get_disorder_vec_of_tested_movern()
	pd.DataFrame({"overall_disorder":OVR_dis, "tested_movern":OVR_tested }).to_feather("%s_overall_test.feather"%(name))
	
	normalizer = mydem.cppdem.get_n_pixels_by_combinations()[0]
	np.save("%s_disorder_normaliser.npy"%(name), normalizer)

	all_disorder = mydem.cppdem.get_best_fits_movern_per_BK()
	np.save("%s_concavity_tot.npy"%(name), all_disorder[0])
	print("Getting results")
	results = np.array(mydem.cppdem.get_all_disorder_values()[0])
	np.save("%s_disorder_tot.npy"%(name), results)
	XY = mydem.cppdem.query_xy_for_each_basin()["0"]
	tdf = pd.DataFrame(XY)
	tdf.to_feather("%s_XY.feather"%(name))



def post_process_basins(prefix = ""):

	print("Done with the calculation, let me preprocess some info about the basins")
	df = pd.read_csv(prefix + "basin_outlets.csv")
	N = list(range(df.shape[0])) # for single analysis
	per = {"X":[], "Y":[], "Z":[], "ID":[], "size" :[]}
	IDcol = {"ID": [],"color": [], "size" : []}
	for i in N:
		name = prefix + "basin_%s"%(i)
		MD = lsd.LSDDEM(file_name = "%s.tif"%(name), already_preprocessed = True)
		# Extracting basins
		MD.CommonFlowRoutines()
		MD.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = 200)
		MD.DefineCatchment( method="from_XY", X_coords = [ float(df["X"][df["ID"] == i].values[0]) ] ,Y_coords = [ float(df["Y"][df["ID"] == i].values[0]) ], coord_search_radius_nodes = 0 ) #, X_coords = [X_coordinates_outlets[7]], Y_coords = [Y_coordinates_outlets[7]])
		MD.GenerateChi(theta = 0.4, A_0 = 1)
		this = MD.cppdem.extract_perimeter_of_basins()
		per["X"].append(this[0]["X"])
		per["Y"].append(this[0]["Y"])
		per["Z"].append(this[0]["Z"])
		per["ID"].append(np.full(this[0]["Z"].shape[0],i))
		# per["color"].append(np.full(this[0]["Z"].shape[0],str(np.random.rand(3,))))
		per["size"].append(np.full(this[0]["Z"].shape[0],MD.df_base_river["x"].shape[0]))
		r = lambda: random.randint(0,255)
		
		this_random_col = '#%02X%02X%02X' % (r(),r(),r())
		IDcol["ID"].append(i)
		IDcol["color"].append(this_random_col)
		IDcol["size"].append(MD.df_base_river["x"].shape[0])



		# per["X"].append(MD.df_base_river["x"])
		# per["Y"].append(MD.df_base_river["y"])
		# per["Z"].append(MD.df_base_river["elevation"])
		# per["ID"].append(np.full(MD.df_base_river["y"].shape[0],i))

	for key,val in per.items():
		per[key] = np.concatenate(val)

	df = pd.DataFrame(per)


	df.to_csv(prefix+"basin_perimeters.csv", index = False)
	df_basin_info = pd.DataFrame(IDcol)

	df_basin_info.to_csv(prefix+"basin_info.csv", index = False)



def disorder_full_basin(chi,elevation, need_sorting = True):
	"""
		This function takes chi and elevation array and returns the Disorder value
		param:
			chi: the chi array
			elevation: the elevation array
			need_sorting: if the array is not sorted by elevation prior to the calculation
		B.G.
	"""
	if(need_sorting):
		index = np.argsort(elevation)
		chi = chi[index]

	return (np.sum(np.abs(chi[1:] - chi[:-1])) - (chi.max() - chi.min()) ) / (chi.max() - chi.min())


def disorder_full_basin_uncert(chi,elevation, n_MC_iteration = 100, proportion_of_points_selected = 0.8):
	"""
		This function takes chi and elevation array and returns the Disorder value
		param:
			chi: the chi array
			elevation: the elevation array
			need_sorting: if the array is not sorted by elevation prior to the calculation
		B.G.
	"""
	index = np.argsort(elevation)
	chi = chi[index]
	all_disorders = []
	for i in range(n_MC_iteration):
		new_indice = np.random.choice(np.arange(chi.shape[0]), round(proportion_of_points_selected * chi.shape[0]), replace=False)
		tchi = chi[np.sort(new_indice)]
		all_disorders.append((np.sum(np.abs(tchi[1:] - tchi[:-1])) - (tchi.max() - tchi.min()) ) / (tchi.max() - tchi.min()))

	return all_disorders