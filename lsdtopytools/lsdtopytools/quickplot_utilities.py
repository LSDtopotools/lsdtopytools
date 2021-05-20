"""
This file contains supporting routines for the quickplot facilities.
For example, determining the right size for a figure or extracting basin outines.
It might send data to cpp at some point, I don't know but it does contains useful small routines I ended up 
using in other project so feel free to pick some in it.
B.G. 19/12/2018
"""
from matplotlib import pyplot as plt
import numpy as np


def create_single_fig_from_width(plotXextent, plotYextent,width = 4, width_units = "inches", background_color = "white"):
	"""
	Creates a figure with an adapted height from its width.
	Arguments:
		plotXextent: the size of the plotted object, to determine the best ratio. For a map, it would be n-columns and n-rows in the raster. For a scatter it would be xmax-xmin and ymax-ymin. ...
		width (float): the required width.
		width_units (str): inches (erk... but matplotlib's norm), or cm/centimetre/centimeter
		background_color (str): background color, can be any color recognised by matplotlib (e.g., HTML color code).
	Returns: 
		A matplotib figure object, hopefully correctly sized
	Authors:
		B.G - 19/12/2018
	"""
	# First I am getting the ratio between Y and X:
	ratioExtent = plotYextent/(plotXextent*1)
	# Easy, now eventually converting the units
	width = width/2.54 if width_units.lower() in ["cm","centimeters","centimeter","centimetre","centimetres"] else width
	return plt.figure(figsize = [width,ratioExtent*width], facecolor = background_color)

def fix_map_axis_to_kms(this_axis, this_fontsize,figure_width):
	"""
		Quick utility that take an axis and reduce its ticks from meters to km.
		Parameters:
			this_axis: the matplotlib axis
			this_fontsize(float): the font size for labels
			figure_width(float): figure width to adapt some text
		Returns:
			Nothing, changes stuff inside the ax
	"""
		# Fixing the ticks: kms rather than metres

	xti = this_axis.get_xticks()
	xtlab = []
	for i in range(len(xti)):
		xtlab.append(str(int(xti[i]/1000)))
	this_axis.set_xticklabels(xtlab, fontsize = this_fontsize)
	yti = this_axis.get_yticks()
	ytlab = []
	for i in range(len(yti)):
		ytlab.append(str(int(yti[i]/1000)))
	this_axis.set_yticklabels(ytlab, fontsize = this_fontsize)

	# labels:
	this_axis.set_xlabel("Easting (km)", fontsize = this_fontsize * 1.2)
	this_axis.set_ylabel("Northing (km)", fontsize = this_fontsize * 1.2)

def finalise_fig(this_fig, these_ax, output, mydem, suffix, format_figure, dpi, force_path = False, path_to_force = ""):
	"""
		When any figure is created with the quickplot engine, user can choose between "finalising" methods:
		The figure can be saved, showed (depending on your matplotlib backend), returned, or leave as it is.
		More details in the argument description.
		Arguments:
			this_fig: The matplotlib figure object
			these_ax: the (single or list of) ax associated with the plotting. Multiple axes are for more complexe plots.
			output (str): the output method. Can be:
				"show": will show the figure with plt.show(). Can not work if your backend is not properly setted.
				"save": will save the figure to a file.
				"return": will return fig, ax for further modifications. For example if you want to add your own scatter on the top of a map, and you just want quickplot to plot the "basemap" basically.
				"Nothing": In some cases, the figure can be passed as argument from a previous manipulation to add the quickplot on top of that and get it back. Quite rare, I needed that for specific cases such as animation plotting. it's there is you need it.
			suffix (str): in case of saving output, this suffix is the name of the fig before the ".". I don't remember why I named it suffix, it made sense back in the days.
			format_figure (str): In case of saving output, this is the format of the figure. It depends on your matplotlib backend, "png" "svg" "pdf" work most of the time.
		Returns:
			Depends on the output describe above
		Authors:
			B.G.
		Date:
			11/2018 (last update on the 23/02/2019)
	"""
	# returning it eventually
	if(output == "show"):
		plt.show()
		plt.clf()
	elif(output == "save"):
		if(force_path):
			plt.savefig(path_to_force+mydem.prefix+"_"+suffix+"."+format_figure, dpi = dpi)
		else:
			plt.savefig(mydem.save_dir+mydem.prefix+"_"+suffix+"."+format_figure, dpi = dpi)
		plt.clf()
	elif(output == "return"):
		return this_fig,these_ax
	elif(output == "nothing"):
		print("You're ouput mode is set to nothing, ain't returning stuff but modify your fig/axes in place.")
	else:
		print("FATALERROR::I did not recognise your output option. Aborting.")
		quit() 

def add_basin_outlines(mydem, fig, ax, size_outline = 1, zorder = 2, color = "k"):
	"""
		Add catchment outlines to any axis given with its associated LSDDEM object ofc.
		No need to mention that catchment needs to be extracted within the LSDDEM object beforehand.
		Arguments:
			fig: the matplotlib figure
			ax: the matplotlib ax
			size_outline: size (in matplotlib units) of the outline
			zorder: the matplotlib zorder to use
			color (any color code compatible with matplotlib): the color of the outline. Default is black
		Returns:
			Nothing, It directly acts on the figure passed in argument
		Authors:
			B.G.
		Date:
			12/2018 (last update on the 23/02/2018)
	"""
	# getting dict of perimeters
	outlines = mydem.cppdem.get_catchment_perimeter()
	# keys are x,y,elevation,basin_key
	for key,val in outlines.items():
		print(key)
		ax.scatter(val["x"], val["y"], c = color, s = size_outline, zorder = zorder, lw =0)

def size_my_points(Iarray, minsize,maxsize):
	"""
		Optimising my point normalisation here. Will be coded once for all.
		Take an array (Initial Array -> Iarray) and returns a normalised version with new bounds.
		Argumetns:
			Iarray(numpy nd_array): The initial array
			minsize (float): the new minimum
			maxsize (float): the new maximum
		Returns:
			The new array
		Authors:
			B.G. and Wikipedia for the normalisation formula
		Date:
			12/2018 (last update: 23/02/2019)
	"""
	# getting min and max
	tmax = np.nanmax(Iarray)
	tmin = np.nanmin(Iarray)
	Iarray = (Iarray-tmin)/(tmax-tmin)
	# Now my data is between 0 and 1
	Iarray = (Iarray * (maxsize-minsize)) + minsize
	# 0_1 -> 0_(maxsize - minsize) -> minsize_maxsize
	return Iarray
