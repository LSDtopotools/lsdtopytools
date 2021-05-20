#!/usr/bin/env python
"""
Command-line tool to automate various csv operations, potentially in a spatial way
So far mostly testing purposes
B.G.
"""
from lsdtopytools import LSDDEM, raster_loader as rl # I am telling python I will need this module to run.
from lsdtopytools import argparser_debug as AGPD # I am telling python I will need this module to run.
from lsdtopytools import quickplot as qp, quickplot_movern as qmn # We will need the plotting routines
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from lsdtopytools import quickplot_utilities as QU
import time as clock # Basic benchmarking
import sys # manage the argv
import glob

def burn_rast_val_to_csv():
	"""
		This scripst burns raster data to a csv containing point coordinate. For example if you want to add lithology ID to river points.
		Usage:
			lsdtt-burn2point file=name_of_raster.tif csv=name_of_csv.csv new_column=burned_data_column_name
	"""
	# Here are the different parameters and their default value fr this script
	default_param = AGPD.get_common_default_param()
	default_param["new_column"] = "burned_data"
	default_param["read_csv"] = "to_burn.csv"
	default_param["save_csv"] = "burned.csv"
	default_param["X_col"] = "X"
	default_param["Y_col"] = "Y"

	default_param = AGPD.ingest_param(default_param, sys.argv)

	if(default_param["help"] or len(sys.argv)==1):
		print("""
			This command-line tool is a command line tool. Documentation to write.
			lsdtt-burn2csv file=example.tif new_column=tectonic_unit read_csv=name_of_csv.csv save_csv=my_new_csv.csv X_col=X Y_col=Y

			""")
		quit()


	print("Welcome to the command-line tool to .")
	print("Let me first load the raster ...")
	mydem = LSDDEM(file_name = default_param["file"], path=default_param["path"], already_preprocessed = True, verbose = False)
	df = pd.read_csv(default_param["read_csv"])
	res = mydem.cppdem.burn_rast_val_to_xy(df[default_param["X_col"]].values,df[default_param["Y_col"]].values)
	df[default_param["new_column"]] = pd.Series(data = res, index = df.index)
	df.to_csv(default_param["save_csv"], index = False)

def csv_conversions():
	"""
		This scripst burns raster data to a csv containing point coordinate. For example if you want to add lithology ID to river points.
		Usage:
			lsdtt-burn2point file=name_of_raster.tif csv=name_of_csv.csv new_column=burned_data_column_name
	"""
	# Different mode usable for the conversion so far
	possible_modes = ["feather2csv", "csv2feather"]

	# Default parameters
	default_param = AGPD.get_common_default_param()
	default_param["mode"] = "feather2csv"
	default_param["all"] = False
	default_param["extension"] = ".feather"

	# Calling the parameter parser ingester
	default_param = AGPD.ingest_param(default_param, sys.argv)

	# Help message
	if(default_param["help"] or len(sys.argv)==1):
		print("""
This command-line tool is a command line tool to convert from and to csv using pandas engine.
Example:
lsdtt-csv-conversion mode=feather2csv file=myfile.feather

Note that you force conversion of all files with an extension with the option all and extension=.feather:
lsdtt-csv-conversion mode=feather2csv all extension=.feather
			""")
		print("Possible conversion mode are:", possible_modes)
		print("""
Different format descriptions:
feather files: Very efficient file structure by apache arrow. Fast I/O, optimised memory, preferred format for the code. However not readable by human.
csv: Most user-friendly format: compatible with excel or any text editor, however much slower to process by softwares and quickly takes space on disk.
			""")
		quit()

	if(default_param["mode"] not in possible_modes):
		print("I did not understand your conversion mode, it needs to be one of the followings:")
		print(possible_modes)
		return 0

	#Starting the list of file to converts
	file_to_convert = []
	# If all, I am getting all the files with a certain extention
	if(default_param["all"]):
		for file in glob.glob(default_param["extension"]):
			file_to_convert.append(file)
	else:
		# Otherwise just the requested files
		file_to_convert.append(default_param["file"])

	##### COnvert feather files (very fast to process by the code but not readable by humans) to csv
	if(default_param["mode"] == "feather2csv"):
		print("converting", file_to_convert,"from feather to csv")
		for file in file_to_convert:
			print("converting file:", file)
			df = pd.read_feather(file)
			df.to_csv(file.split(".")[:-1]+".csv", index = False)
	##### convert csv to feather
	if(default_param["mode"] == "csv2feather"):
		for file in file_to_convert:
			print("converting file:", file)
			df = pd.read_csv(file)
			df.to_feather(file.split(".")[:-1]+".feather")



