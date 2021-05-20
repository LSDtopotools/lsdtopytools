"""
This class deals with loading raster informations
Authors: B.G.
"""
import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds


def load_raster(fname):
	"""
	Load a raster array with different options. It uses rasterio that itself uses gdal.
	Arguments:
		fname (str): raster to load (path+file_name+format)
	Returns:
		A python dictionnary containing the following "key" -> val:
			"res" -> Resolution of the DEM
			"ncols" -> number of columns
			"nrows" -> number of rows
			"x_min" -> well x minimum
			"y_min" -> and y minimum
			"x_max" -> and x maximum
			"y_max" -> and x maximum
			"extent" -> extent combined in order to match matplotlib
			"array" -> numpy 2D array containing the data
			"crs" -> The crs string (geolocalisation)
			"nodata" -> list of nodata values
	Authors:
		B.G.
	Date:
		23/02/2019
	"""

	# Loading the raster with rasterio
	this_raster = rio.open(fname)

	# Initialising a dictionary containing the raster info for output
	out = {}
	# I am padding a no_data contour
	gt = this_raster.res
	out['res'] = gt[0]
	out["ncols"] = this_raster.width
	out["nrows"] = this_raster.height
	out["x_min"] = this_raster.bounds[0]
	out["y_min"] = this_raster.bounds[1]
	out["x_max"] = this_raster.bounds[2]
	out["y_max"] = this_raster.bounds[3]
	corr = out['res']*2
	out["extent"] = [out["x_min"],out["x_max"]-corr,out["y_min"],out["y_max"]-corr]
	out["array"] = this_raster.read(1)
	try:
		out['crs'] = this_raster.crs['init']
	except (TypeError, KeyError) as e:
		out['crs'] = u'epsg:32601'
	out['nodata'] = this_raster.nodatavals

	
	
	# pixelSizeY =-gt[4]
	# (left=358485.0, bottom=4028985.0, right=590415.0, top=4265115.0)

	return out

def save_raster(Z,x_min,x_max,y_min,y_max,res,crs,fname, fmt = 'GTIFF'):
	'''
		Save a raster at fname. I followed tutorial there to do so : https://rasterio.readthedocs.io/en/latest/quickstart.html#creating-data
		Arguments:
			Z (2d numpy array): the array
			x_min (float): x min
			y_min (float): y min
			x_max (float): x max
			y_max (float): y max
			res (float): resolution
			crs: coordinate system code (usually you will call it from a dem object, just give it dem.crs)
			fname: path+name+.tif according to your OS  ex: Windows: C://Data/Albania/Holtas/Holt.tif, Linux: /home/Henri/Albania/Shenalvash/She_zoom.tif
			fmt (str): string defining the format. "GTIFF" for tif files, see GDAL for option. WARNING: few outputs are buggy, FOR EXAMPLE "ENVI" bil format can be with the wrong dimensions.
		Returns:
			Nothing but saves a raster with the given parameters
		Authors: 
			B.G. 
		Date:
			08/2018
	'''

	transform = from_bounds(x_min, y_max, x_max, y_min, Z.shape[1], Z.shape[0])
	new_dataset = rio.open(fname, 'w', driver=fmt, height=Z.shape[0], width=Z.shape[1], count=1, dtype=Z.dtype.type, crs=crs, transform=transform, nodata = -9999)
	new_dataset.write(Z, 1)
	new_dataset.close()