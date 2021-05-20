"""
Quick visualisation routines for ksn analysis and knickpoint extraction anaysis.
B.G
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import lsdtopytools.quickplot_utilities as QUT, lsdtopytools.quickplot as QP
from lsdtopytools import LSDDEM


def plot_ksn_map(mydem ,figure_width = 4, figure_width_units = "inches", cmap = "RdBu_r", hillshade = True, 
	alpha_hillshade = 0.45, color_min = None, color_max = None, dpi = 300, output = "save",
	 format_figure = "png", min_point_size = 0.5, max_point_size = 3, ksn_min = None, ksn_max = None, ksn_colormod = "default", size_outline = 1, fontsize_ticks =6, fontsize_label = 8):
	"""
		Plot ksn on hillshade or black background. Obviously, you need to calculate it beforeend!
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
			min_point_size (float): minimum size for the point scaled by Drainage area.
			max_point_size (float): maximum size for the point scaled by Drainage area.
			ksn_min (float): k_sn value that defines the minimum value of the color bar (all the ksn below that value will have the minimum color). Can be percentile or absolute value.
			ksn_max (float): k_sn value that defines the maximum value of the color bar (all the ksn above that value will have the maximum color). Can be percentile or absolute value.
			ksn_colormod (str): "default" to scale colormap per value, "percentile" tp scale using stats.
			size_outline (float): size of the basin outline. 0 for no outline. hacky but simple innit?
			fontsize_ticks (float): Font size of the ticks. Might be a bit random, matplotlib can be difficult to tame.
			fontsize_label (float): Font size of the labels. Might be a bit random, matplotlib can be difficult to tame.
		Returns:
			Depends on the output parameters
		Authors:
			B.G
		Date:
			23/02/2019
	"""
	if(hillshade):
		fig, ax1 = QP.plot_nice_topography(mydem ,figure_width = figure_width, figure_width_units = figure_width_units, hillshade = hillshade, 
	alpha_hillshade = 1, color_min = None, color_max = None, dpi = 300, output = "return")
	else:
		# First creating the figure and axis
		fig = QUT.create_single_fig_from_width(mydem.ncols, mydem.nrows, width = figure_width, width_units = figure_width_units, background_color = "white")
		gs = plt.GridSpec(100,100,bottom=0.15, left=0.20, right=0.98, top=0.98)
		ax1 = fig.add_subplot(gs[0:100,0:100], facecolor = "none")
		ax1.imshow(mydem.cppdem.get_PP_raster(), mydem.extent, cmap = "gray", vmin = 10000000, vmax = 10000001, zorder = 1, fontsize_ticks =fontsize_ticks, fontsize_label = fontsize_ticks) # Forcing black background
	
	# Getting basin outlines
	QUT.add_basin_outlines(mydem, fig, ax1, size_outline = size_outline)

	# colour bounds if not manually chosen
	if(ksn_colormod.lower() == "default"):
		ksn_min = mydem.df_ksn["m_chi"].min() if(ksn_min is None) else ksn_min
		ksn_max = mydem.df_ksn["m_chi"].max() if(ksn_max is None) else ksn_max
	if(ksn_colormod.lower() == "percentile"):
		ksn_min = mydem.df_ksn["m_chi"].quantile(0.10) if(ksn_min is None) else mydem.df_ksn["m_chi"].quantile(ksn_min)
		ksn_max = mydem.df_ksn["m_chi"].quantile(0.90) if(ksn_max is None) else mydem.df_ksn["m_chi"].quantile(ksn_max)

	# Plotting ksn
	cb = ax1.scatter(mydem.df_ksn["x"], mydem.df_ksn["y"], s= QUT.size_my_points(mydem.df_ksn["drainage_area"].values, min_point_size, max_point_size),
		c=mydem.df_ksn["m_chi"].values, cmap = cmap, vmin = ksn_min, vmax = ksn_max, zorder = 5, lw = 0) # done

	# Colorbar
	cax = plt.colorbar(cb)
	cax.ax.tick_params(labelsize = fontsize_ticks)
	cax.ax.set_ylabel(r"$k_{sn}$", fontsize = fontsize_label, rotation = -270)


	#Done?
	return QUT.finalise_fig(fig, ax1, output, mydem, "ksn_map", format_figure, dpi)