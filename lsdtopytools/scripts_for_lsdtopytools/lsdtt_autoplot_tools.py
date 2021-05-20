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
import glob
import multiprocessing
import subprocess
# from itertools import product

def load_n_plot(name):
	MD = LSDDEM(file_name = name, already_preprocessed = True)
	qp.plot_nice_topography(MD ,figure_width = 4, figure_width_units = "inches", cmap = "gist_earth", hillshade = True, 
alpha_hillshade = 0.45, color_min = None, color_max = None, dpi = 300, output = "save", format_figure = "png", fontsize_ticks =6, fontsize_label = 8, 
hillshade_cmin = 0, hillshade_cmax = 250,colorbar = False, 
fig = None, ax = None, colorbar_label = None, colorbar_ax = None, force_path = True, path_to_force = "./")
	term = MD.path+MD.prefix+"/"
	subprocess.run("rm -r %s"%(term), shell = True)

def plot_all_tif_of_folder():


	
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["n_threads"] = 4
	default_param = AGPD.ingest_param(default_param, sys.argv)

	file_to_process = []
	suffix = []
	cpt = 1
	for file in glob.glob("*.tif"):
		file_to_process.append( (file,) )
		name = str(cpt)
		cpt+=1
	for file in glob.glob("*.tiff"):
		file_to_process.append( (file,) )
	if(len(file_to_process)==0):
		print("I did not find any file! Aborting.")
		quit()

	with multiprocessing.Pool(processes=int(default_param["n_threads"])) as pool:
		pool.starmap(load_n_plot, file_to_process)

	print("Done with all the rasters")