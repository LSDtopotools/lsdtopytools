"""
This scripts contains global usefeful utilities. No general rules on these, the aim is to decongest the main object from small functions.
B.G.
"""
import pandas as pd


def save_to_database(filename,key, df, metadata = {}):
	"""
	Drops a value to the hdf5 database.
	df is the dataframe containing the data, and kward is the dictionary of arguments
	For example: the df with x/y/chi/... and metadata is the parameter you used.
	Arguments:
		filename (str): path+path+name of the hdf 
		key(str): The key used to save the data. Carefull, if the key exists it will be replaced.
		df(pandas DataFrame): the dataframe to save
		metadata(dict): Optional extra signe arguments: for example {"theta": 0.35,"preprocessing": "carving"}
	B.G. 13/01/2019
	"""
	# Opening the dataframe
	store = pd.HDFStore(filename)
	# Feeding the dataframe, 't' means table format (slightly slower but can be modified)
	store.put(key, df, format="t")
	# feeding the metadata
	store.get_storer(key).attrs.metadata = metadata
	# /!\ Important to properly close the file
	store.close()


def load_from_database(filename,key):
	"""
	Get a dataframe from the hdf5 file and its metadata
	Argument:
		Filename(str): the name of the file to load
		key(str): The key to load: all the different dataframses are stored with a key
	returns:
		Dataframe and dictionary of metadata
	B.G. 13/01/2019
	"""
	# Opening file
	store = pd.HDFStore(filename)
	# getting the df
	data = store[key]
	# And its metadata
	metadata = store.get_storer(key).attrs.metadata
	store.close()
	# Ok returning the data now
	return data, metadata

def load_metadata_from_database(filename,key):
	"""
	Get a dataframe from the hdf5 file and its metadata
	Argument:
		Filename(str): the name of the file to load
		key(str): The key to load: all the different dataframses are stored with a key
	returns:
		Dataframe and dictionary of metadata
	B.G. 13/01/2019
	"""
	# Opening file
	store = pd.HDFStore(filename)
	metadata = store.get_storer(key).attrs.metadata
	store.close()
	# Ok returning the data now
	return metadata

def trim_knickpoint_internal(df,method,trimmer):
	"""
		Internal function used to trim the knickpoint dataset.
		I put that here to unload the main lsdtopytools, however, I don't recommend using that function
		for any general purpose as too specific.
		See Mother Function in lsdtopytools.LSDDEM.trim_knickpoint_dataset(...) for doc
		Authors:
			B.G.
		Date:
			23/02/2018
	"""

	if(method=="third_quartile"):
		df1 = df["delta_ksn"][df["delta_ksn"]>0] # selecting only the positive kps
		df1 = df1["delta_ksn"][df1["delta_ksn"]>=df1["delta_ksn"].quartile(75)] # getting the third_quartile
		df2 = df["delta_ksn"][df["delta_ksn"]<0] # selecting only the negative kps
		df2 = df2["delta_ksn"][df2["delta_ksn"]>=df2["delta_ksn"].quartile(75)] # getting the third_quartile
		df3 = df["delta_segelev"][df["delta_segelev"]>0] # selecting only the positive kps
		df3 = df3["delta_segelev"][df3["delta_segelev"]>=df3["delta_segelev"].quartile(75)] # getting the third_quartile
	elif(method=="percentile_from_dict"):
		df1 = df["delta_ksn"][df["delta_ksn"]>0] # selecting only the positive kps
		df1 = df1["delta_ksn"][df1["delta_ksn"]>=df1["delta_ksn"].quartile(trimmer["percentile_cut_positive_slope_break"])] # getting the third_quartile
		df2 = df["delta_ksn"][df["delta_ksn"]<0] # selecting only the negative kps
		df2 = df2["delta_ksn"][df2["delta_ksn"]>=df2["delta_ksn"].quartile(trimmer["percentile_cut_negative_slope_break"])] # getting the third_quartile
		df3 = df["delta_segelev"][df["delta_segelev"]>0] # selecting only the positive kps
		df3 = df3["delta_segelev"][df3["delta_segelev"]>=df3["delta_segelev"].quartile(trimmer["percentile_cut_step"])] # getting the third_quartile
	elif(method=="val_from_dict"):
		df1 = df["delta_ksn"][df["delta_ksn"]>0] # selecting only the positive kps
		df1 = df1["delta_ksn"][df1["delta_ksn"]>=df1["delta_ksn"].quartile(trimmer["value_cut_positive_slope_break"])] # getting the third_quartile
		df2 = df["delta_ksn"][df["delta_ksn"]<0] # selecting only the negative kps
		df2 = df2["delta_ksn"][df2["delta_ksn"]>=df2["delta_ksn"].quartile(trimmer["value_cut_negative_slope_break"])] # getting the third_quartile
		df3 = df["delta_segelev"][df["delta_segelev"]>0] # selecting only the positive kps
		df3 = df3["delta_segelev"][df3["delta_segelev"]>=df3["delta_segelev"].quartile(trimmer["value_cut_step"])] # getting the third_quartile

	return pd.concat([df1,df2,df3], reset_index = True)






















class WrongInputTypeError(Exception):
	"""Raised when the input value is too large"""
	pass