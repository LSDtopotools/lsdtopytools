"""
This file contains plotting routines organised in independent functions.
I don't know how I will separate these in the future, but quickplot aims to provide very basic first order plots
All the routines should leave 3 possible ouputs: save the figure, get the figure (for further modifications) and show the figure.
B.G. 2018
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import lsdtopytools.quickplot_utilities as QUT
from lsdtopytools import LSDDEM

def numfmt(x, pos):
	"""
	Plotting subfunction to automate tick formatting from metres to kilometres
	B.G
	"""
	s = '{:d}'.format(int(round(x / 1000.0)))
	return s
import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting

def plot_nice_topography(mydem ,figure_width = 4, figure_width_units = "inches", cmap = "gist_earth", hillshade = True, 
	alpha_hillshade = 0.45, color_min = None, color_max = None, dpi = 300, output = "save", format_figure = "png", fontsize_ticks =6, fontsize_label = 8, 
	hillshade_cmin = 0, hillshade_cmax = 250,colorbar = False, 
	fig = None, ax = None, colorbar_label = None, colorbar_ax = None, 
	xlim = [],ylim = [],**kwargs):
	"""
		Plot a nice topographic map. You can adjust the following set of parameters to customise it:
		Arguments:
			mydem: the LSDDEM object. Required.
			figure_width (float): figure width in inches (Default value because of matplotlib...) or centimeters.
			figure_width_units (str): "inches" (Default) or "centimetres"
			cmap (str): the name of the colormap to use, see https://matplotlib.org/examples/color/colormaps_reference.html for list.
			hillshade (bool): Drape an hillshade on the top of the figure
			alpha_hillshade (float): regulate the transparency of the hillshade. Between 0 (transparent) and 1 (opaque).
			color_min (float): min elevation corresponding to the min color on the colormap. None = auto.
			color_max (float): see color_min and replace min by max.
			dpi (int): Figure quality, increase will increase the figure quality and its size ofc.
			output (str): "save" to save figure, "return" to return fig/axis, "show" to show.
			format_figure (str): the extension of the saved bitmap: "png", "jpg", "svg" reccomended. For the rest see: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
			fontsize_ticks (float): the fontsize of the ticks
			fontsize_label (float): font size of the labels
			fig (matplotlib figure): Default None, in certain cases you may want to add teh basemap plot on the top of an existing. 
			ax (Matplotlib axis): Default None. if custom Figure passed as argument, you also need to pass an axis
			colorbar_label (str): label of the colorbar if activated.
			colorbar_ax (mpl axis): the axis where to place the colorbar (Default will be auto no worries)
		Returns:
			Depends on parameters
		Authors:
			Boris Gailleton
		Date:
			19/12/2018 - last update 09/03/2019
	"""

	# First creating the figure and axis
	if(fig is None):
		fig = QUT.create_single_fig_from_width(mydem.ncols, mydem.nrows, width = figure_width, width_units = figure_width_units, background_color = "white")
	gs = plt.GridSpec(100,100,bottom=0.15, left=0.20, right=0.98, top=0.98)
	if(ax is None):
		ax1 = fig.add_subplot(gs[0:100,0:100], facecolor = "none")
	else:
		ax1 = ax

	# Plotting the figure
	cb = ax1.imshow(mydem.cppdem.get_PP_raster(), extent = mydem.extent, interpolation = "none", cmap = cmap, vmin= color_min, vmax = color_max, zorder = 1)
	if( (hillshade_cmin is not None) and (hillshade_cmin is not None)):
		cb2 = ax1.imshow(mydem.get_hillshade(), extent = mydem.extent, interpolation = "none", cmap = "gray", vmin= hillshade_cmin, vmax = hillshade_cmax, alpha = alpha_hillshade, zorder = 2)
	else:
		arr = mydem.get_hillshade()
		arr[arr==-9999] = np.nan
		cb2 = ax1.imshow(arr, extent = mydem.extent, interpolation = "none", cmap = "gray", alpha = alpha_hillshade, zorder = 2)
	if(colorbar):
		if(colorbar_ax is None):
			cbar = plt.colorbar(cb)
		else:
			cbar = plt.colorbar(cb,colorbar_ax)

		cbar.ax.tick_params(labelsize=fontsize_ticks)

		if(~(colorbar_label is None)):
			cbar.ax.set_ylabel(colorbar_label)

	# fixing map lims
	if(len(xlim) == 2):
		ax1.set_xlim(xlim[0],xlim[1])
	if(len(ylim) == 2):
		ax1.set_ylim(ylim[0],ylim[1])

	# Fixing the ticks: kms rather than metres
	QUT.fix_map_axis_to_kms(ax1, fontsize_ticks, figure_width)
	# labels:
	ax1.set_xlabel("Easting (km)", fontsize = fontsize_label)
	ax1.set_ylabel("Northing (km)", fontsize = fontsize_label)
	if("force_path" in kwargs and "path_to_force" in kwargs):
		return QUT.finalise_fig(fig, ax1, output, mydem, "_topo.", format_figure, dpi, force_path = kwargs["force_path"], path_to_force = kwargs["path_to_force"])
	else:
		return QUT.finalise_fig(fig, ax1, output, mydem, "_topo.", format_figure, dpi)


def get_basemap(mydem , figsize = (4,5), cmap = "gist_earth", hillshade = True, 
	alpha_hillshade = 0.45, cmin = None, cmax = None,
	hillshade_cmin = 0, hillshade_cmax = 1, colorbar = False, 
	fig = None, ax = None, colorbar_label = None, colorbar_ax = None, fontsize_ticks = 8, normalise_HS = True):
	"""
		Plot a nice topographic map. You can adjust the following set of parameters to customise it:
		Arguments:
			mydem: the LSDDEM object. Required.
			figure_width (float): figure width in inches (Default value because of matplotlib...) or centimeters.
			figure_width_units (str): "inches" (Default) or "centimetres"
			cmap (str): the name of the colormap to use, see https://matplotlib.org/examples/color/colormaps_reference.html for list.
			hillshade (bool): Drape an hillshade on the top of the figure
			alpha_hillshade (float): regulate the transparency of the hillshade. Between 0 (transparent) and 1 (opaque).
			cmin (float): min elevation corresponding to the min color on the colormap. None = auto.
			cmax (float): see cmin and replace min by max.
			dpi (int): Figure quality, increase will increase the figure quality and its size ofc.
			output (str): "save" to save figure, "return" to return fig/axis, "show" to show.
			format_figure (str): the extension of the saved bitmap: "png", "jpg", "svg" reccomended. For the rest see: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
			fontsize_ticks (float): the fontsize of the ticks
			fontsize_label (float): font size of the labels
			fig (matplotlib figure): Default None, in certain cases you may want to add teh basemap plot on the top of an existing. 
			ax (Matplotlib axis): Default None. if custom Figure passed as argument, you also need to pass an axis
			colorbar_label (str): label of the colorbar if activated.
			colorbar_ax (mpl axis): the axis where to place the colorbar (Default will be auto no worries)
		Returns:
			Depends on parameters
		Authors:
			Boris Gailleton
		Date:
			19/12/2018 - last update 09/03/2019
	"""

	# First creating the figure and axis
	if(fig is None):
		fig = plt.figure(figsize= figsize)
	gs = plt.GridSpec(100,100,bottom=0.15, left=0.20, right=0.98, top=0.98)
	if(ax is None):
		ax1 = fig.add_subplot(gs[0:100,0:100], facecolor = "none")
	else:
		ax1 = ax

	toplot = mydem.cppdem.get_PP_raster()

	if(cmin is None):
		cmin = np.nanmin(toplot[toplot>=0])

	# Plotting the figure
	cb = ax1.imshow(toplot, extent = mydem.extent, interpolation = "none", cmap = cmap, vmin= cmin, vmax = cmax, zorder = 0.5)
	if(hillshade):
		arr = mydem.get_hillshade()
		arr[arr==-9999] = np.nan
		if(normalise_HS):
			arr = arr/np.nanmax(arr)
		if( (hillshade_cmin is not None) and (hillshade_cmin is not None)):
			cb2 = ax1.imshow(arr, extent = mydem.extent, interpolation = "none", cmap = "gray", vmin= hillshade_cmin, vmax = hillshade_cmax, alpha = alpha_hillshade, zorder = 0.8)
		else:

			cb2 = ax1.imshow(arr, extent = mydem.extent, interpolation = "none", cmap = "gray", alpha = alpha_hillshade, zorder = 0.8, vmin= np.nanpercentile(arr,10), vmax = np.nanpercentile(arr,90))
		if(colorbar):
			if(colorbar_ax is None):
				cbar = plt.colorbar(cb)
			else:
				cbar = plt.colorbar(cb,colorbar_ax)

			if(~(colorbar_label is None)):
				cbar.ax.set_ylabel(colorbar_label)
		
	xfmt = tkr.FuncFormatter(numfmt)
	ax1.xaxis.set_major_formatter(xfmt)
	ax1.yaxis.set_major_formatter(xfmt)
	# labels:
	ax1.set_xlabel("Easting (km)")
	ax1.set_ylabel("Northing (km)")
	return fig, ax1


def plot_preprocessing_diff(mydem ,figure_width = 4, figure_width_units = "inches", cmap = "RdBu_r", this_fontsize = 6,
	color_min = None, color_max = None, dpi = 300, output = "save", format_figure = "png", n_dec_lab = 5):
	"""
		Plot a differential raster between carving/breaching/filling and the original data. Useful to check preprocessing artifacts.
		Arguments:
			mydem: the LSDDEM object. Required.
			figure_width (float): figure width in inches (Default value because of matplotlib...) or centimeters.
			figure_width_units (str): "inches" (Default) or "centimetres"
			cmap (str): the name of the colormap to use, see https://matplotlib.org/examples/color/colormaps_reference.html for list.
			color_min (float): min elevation corresponding to the min color on the colormap. None = auto.
			color_max (float): see color_min and replace min by max.
			dpi (int): Figure quality, increase will increase the figure quality and its size ofc.
			output (str): "save" to save figure, "return" to return fig/axis, "show" to show.
			format_figure (str): the extension of the saved bitmap: "png", "jpg", "svg" reccomended. For the rest see: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
			n_dec_lab (int): A bit experimental, it manage the precision of colorbar labels.
			this_fontsize (float): fontsize of the minor labels
		Returns:
			Depends on parameters
		Authors:
			Boris Gailleton
		Date:
			19/12/2018
	"""

	# First creating the figure and axis
	fig = QUT.create_single_fig_from_width(mydem.ncols, mydem.nrows, width = figure_width, width_units = figure_width_units, background_color = "white")
	gs = plt.GridSpec(100,100,bottom=0.15, left=0.18, right=0.98, top=0.98)
	ax1 = fig.add_subplot(gs[0:100,0:100], facecolor = "none")

	# Get the differential raster
	pre = mydem.cppdem.get_base_raster()
	post = mydem.cppdem.get_PP_raster()
	pre[pre==-9999] = np.nan
	post[post==-9999] = np.nan
	diff = post - pre


	#Dealing wih the colorbar
	if(color_min is None):
		this_min = np.nanmin(diff)
	else:
		this_min = color_min
	if(color_max is None):
		this_max = np.nanmax(diff)
	else:
		this_max = color_max

	# Plotting the figure
	has_been_plotted = False
	is_min_cax = False
	if(this_max>0):
		has_been_plotted = True
		tdiff = np.copy(diff)
		tdiff[diff<0] = np.nan
		cb = ax1.imshow(tdiff, extent = mydem.extent, interpolation = "none", cmap = cmap, vmin = -this_max, vmax = this_max)
		del tdiff
	if(this_min<0):
		is_min_cax = True
		has_been_plotted = True
		tdiff = np.copy(diff)
		tdiff[tdiff>0] = np.nan
		cb = ax1.imshow(tdiff, extent = mydem.extent, interpolation = "none", cmap = cmap, vmin = this_min, vmax = -this_min)
		del tdiff


	# Check
	if(has_been_plotted == False):
		print("Not a single change in the preprocessing, Aborting the plot, it will be an empty plot")
	else:
		cax = plt.colorbar(cb )
		cax.ax.set_ylabel(r"$\Delta$ elevation", fontsize = this_fontsize, rotation = -270)
		cax.ax.get_yaxis().labelpad = -10
		extr = cax.get_ticks()
		if(is_min_cax):
			cax.set_ticks([this_min,0,-this_min])
		if(is_min_cax== False):
			cax.set_ticks([-this_max,0,this_max])

		cax.set_ticklabels([str(this_min)[:n_dec_lab],0,str(this_max)[:n_dec_lab]])
		cax.ax.tick_params(labelsize = this_fontsize) 



	# Fixing the ticks: kms rather than metres
	xti = ax1.get_xticks()
	xtlab = []
	for i in range(len(xti)):
		xtlab.append(str(xti[i]/1000))
	ax1.set_xticklabels(xtlab, fontsize = this_fontsize)
	yti = ax1.get_yticks()
	ytlab = []
	for i in range(len(yti)):
		ytlab.append(str(yti[i]/1000))
	ax1.set_yticklabels(ytlab, fontsize = this_fontsize)

	# labels:
	ax1.set_xlabel("Easting (km)", fontsize = this_fontsize * 1.2)
	ax1.set_ylabel("Northing (km)", fontsize = this_fontsize * 1.2)


	# returning it eventually
	if(output == "show"):
		plt.show()
		plt.clf()
	elif(output == "save"):
		# Dealing with the particle
		if(mydem.has_been_filled_by_me and mydem.has_been_carved_by_me):
			part = "PP_diff."
		elif(mydem.has_been_carved_by_me):
			part = "only_carved_diff."
		elif(mydem.has_been_filled_by_me):
			part = "only_filled_diff."
		# Saving the figure to the disk
		plt.savefig(mydem.save_dir+mydem.prefix+part+format_figure, dpi = dpi)

	elif(output == "return"):
		return fig,ax
	else:
		print("FATALERROR::I did not recognise your output option. Aborting.")
		quit()






def plot_check_catchments(mydem ,figure_width = 4, figure_width_units = "inches", cmap = "jet", alpha_hillshade = 0.95, 
	this_fontsize = 6, alpha_catchments = 0.75,  dpi = 300, output = "save", format_figure = "png"):
	"""
		Plot to check basin selection. You can adjust the following set of parameters to customise it:
		Arguments:
			mydem: the LSDDEM object. Required.
			figure_width (float): figure width in inches (Default value because of matplotlib...) or centimeters.
			figure_width_units (str): "inches" (Default) or "centimetres"
			cmap (str): the name of the colormap to use, see https://matplotlib.org/examples/color/colormaps_reference.html for list.
			alpha_hillshade (float): regulate the transparency of the background hillshade. Between 0 (transparent) and 1 (opaque).
			dpi (int): Figure quality, increase will increase the figure quality and its size ofc.
			output (str): "save" to save figure, "return" to return fig/axis, "show" to show.
			format_figure (str): the extension of the saved bitmap: "png", "jpg", "svg" reccomended. For the rest see: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
			this_fontsize (flaot): font size of minor texts
			alpha_catchments: transparency of basin colors.
		Returns:
			Depends on parameters
		Authors:
			Boris Gailleton
		Date:
			19/12/2018
	"""

	# First creating the figure and axis
	fig = QUT.create_single_fig_from_width(mydem.ncols, mydem.nrows, width = figure_width, width_units = figure_width_units, background_color = "white")
	gs = plt.GridSpec(100,100,bottom=0.15, left=0.20, right=0.98, top=0.98)
	ax1 = fig.add_subplot(gs[0:100,0:100], facecolor = "none")

	# Plotting the figure
	ax1.imshow(mydem.get_hillshade(), extent = mydem.extent, interpolation = "none", cmap = "gray", vmin= 0, vmax = 255, alpha = alpha_hillshade)

	# Getting the catchment data
	BAS = mydem.cppdem.get_chi_basin().astype(np.float32)# switching to float32 to enable nan support for transparency
	BAS[BAS==-9999] = np.nan
	cb = ax1.imshow(BAS, extent = mydem.extent, interpolation = "none", cmap = cmap, vmin= 0, alpha = alpha_catchments)
	cax = plt.colorbar(cb )
	cax.ax.set_ylabel("Basin Keys", fontsize = this_fontsize, rotation = -270)
	cax.ax.get_yaxis().labelpad = -10
	cax.ax.tick_params(labelsize = this_fontsize) 
	QUT.fix_map_axis_to_kms(ax1, this_fontsize, figure_width)
	return QUT.finalise_fig(fig, ax1, output, mydem, "catchment_check", format_figure, dpi)