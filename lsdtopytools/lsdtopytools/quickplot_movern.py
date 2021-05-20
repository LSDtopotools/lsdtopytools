"""
Quick visualisation routines for m_over_n analysis.
B.G
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import lsdtopytools.quickplot_utilities as QUT
from lsdtopytools import LSDDEM

def plot_disorder_results(mydem, normalise = False, figsize = (4,3), dpi = 300, output = "save", format_figure = "png", legend = True,
 cumulative_best_fit = True):
	"""
		Plot results from simple disorder analysis (Mudd et al., 2018).
		The value represents the scatter of chi-elevation plots for each basins: High value = more scatter, low value = less scatter.
		Note that if you have basins of different area, their value is harder to compare, but you can normalise it to the maximum.
		Arguments:
			mydem: an LSDDEM object where the disorder best-fit has been calculated
			normalise (bool): False to plot the absolute disorder value, True to normalise to maximum (0-1). You want to normalise if you need to actually compare several basins as the disorder vallue is linked to the basin size.
			figsize (tuple or list of 2): width, heigth in matplotlib units (inches I guess).
			dpi (int): figure quality if saveing ouput is chosen
			output (str): "save" to save figure, "return" to return fig/axis, "show" to show.
			legend (bool): Do you need to plot the legend. True or False.
			cumulative_best_fit (bool): Add something to the plot.
		Returns:
			Depends on the output size			
		Authors:
			Boris Gailleton
		Date:
			04/01/2019
	"""
	# Preparing figure
	fig = plt.figure(figsize = figsize)
	gs = plt.GridSpec(100,100,bottom=0.15, left=0.18, right=0.87, top=0.98)
	ax1 = fig.add_subplot(gs[0:100,0:100], facecolor = "none") # Plot the movern scatter and lien

	# Getting the data
	X_movern = mydem.cppdem.get_disorder_vec_of_tested_movern()
	dict_per_basins = mydem.cppdem.get_disorder_dict()

	best_fit = np.zeros(len(X_movern))

	# Plotting it
	for key,val in dict_per_basins.items():
		tval = np.asarray(val)
		tval = tval/np.nanmax(tval) if (normalise) else tval
		# Plotting the line
		ax1.plot(X_movern, tval, lw =1, alpha = 0.8, zorder = 1, label = key)
		ax1.scatter(X_movern, tval, lw =1, s = 5, c = "k", zorder = 2, marker = "+")
		best_fit[np.argmin(tval)] += 1 # Incrementing the cumulative best-fit

	if(cumulative_best_fit):
		ax2 = ax1.twinx() # plot the cumulative best fit
		ax2.plot(X_movern, best_fit, color = "k", lw = 1, ls = "-.", zorder = 0.5)
		ax2.fill_between(X_movern,0,best_fit,lw =0, color = "k", alpha = 0.3333)
		ax2.set_ylabel("Best-fit cumulative")

	ax1.legend() if (legend) else 0
	ax1.set_xlabel(r"$\theta$ tested")
	ylab = "Disorder"
	ylab += " (norm.)" if(normalise) else ""
	ax1.set_ylabel(ylab)

	return QUT.finalise_fig(fig, ax1, output, mydem, "concavity_disorder_results", format_figure, dpi)

def plot_disorder_map(mydem ,figure_width = 4, figure_width_units = "inches", cmap = "jet", alpha_hillshade = 0.95, 
	this_fontsize = 6, alpha_catchments = 0.75,  dpi = 300, output = "save", format_figure = "png"):
	"""
		Plot a map of best fit theta using disorder method. You can adjust the following set of parameters to customise it:
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

	# Getting the movern data
	# Getting the data
	X_movern = mydem.cppdem.get_disorder_vec_of_tested_movern()
	dict_per_basins = mydem.cppdem.get_disorder_dict()

	best_fit = np.array(X_movern)

	# Plotting it
	for key,val in dict_per_basins.items():
		tval = np.asarray(val)
		bf = best_fit[np.argmin(tval)] # Incrementing the cumulative best-fit
		BAS[BAS == key] = bf


	cb = ax1.imshow(BAS, extent = mydem.extent, interpolation = "none", cmap = cmap, alpha = alpha_catchments)
	cax = plt.colorbar(cb )
	cax.ax.set_ylabel(r"Best fit $\theta$", fontsize = this_fontsize, rotation = -270)
	# cax.ax.get_yaxis().labelpad = -10
	cax.ax.tick_params(labelsize = this_fontsize) 
	QUT.fix_map_axis_to_kms(ax1, this_fontsize, figure_width)
	return QUT.finalise_fig(fig, ax1, output, mydem, "concavity_disorder_map", format_figure, dpi)