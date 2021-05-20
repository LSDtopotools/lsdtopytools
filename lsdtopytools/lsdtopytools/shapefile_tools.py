"""
This files will manage all the shape-file related operation. Espescially I/O probably.

Authors: B.G
"""
import fiona, shapely as shp, geopandas as gpd
import numpy as np
import pandas as pd

class lsdSHP(object):
	"""
		lsdSHP class contains tools to interact with shapefiles and make them ingestable to 
	"""
	def __init__(self, file_name, path = "./", verbose = True):
		"""
			Constructor of the lsdSHP object.
			Arguments:
				file_name (str): the name of the file including the main extention (e.g. "gabulan.shp", "falafel.json")
				path (str): path to the file to read/write (default=current folder)
				verbose (bool): The code is quite chatty, you may want to ask it to keep it quiet.
			Returns:
				a brand new lsdSHP object Yeah
			Authors:
				B.G.
			Date:
				04/03/2019
		"""
		# Ignore that line
		super(lsdSHP, self).__init__()
		# File to load
		self.file_name = file_name
		# and its path
		self.path = path

		# Loading the file with geopandas
		self.shpdf = gpd.read_file(path+self.file_name)

		# Loading a bunch of useful info to have
		self.data_info = {"type": self.shpdf.geometry.iloc[0], "n_elements": self.shpdf}

		# Verbosity of the code
		self.verbose = verbose

		# Getting the prefix for saving all your files:
		self.prefix = str("".join(self.file_name.split(".")[:-1])) # -> prefix = base name without extention




		
	def line_to_points(self, interval = 30, save_to_shapefile = True, save_to_csv = False):
		"""
			Break each line instance of the shapefile into points.
			Arguments:
				interval (float): Interval between each points (in map units, preferably meters)
				save_to_shapefile (bool): if True, save a shapefile with the new points
				save_to_csv (bool): if True, save a csv with 3 columns (line_ID, X, Y)
			Returns:
				a Pandas dataframe with 3 columns (line_ID, X, Y)
			Authors:
				B.G.
			Date:
				04/03/2019
		"""

		print("I am trying to break your line, this feature hasn't been widely tested yet, please let me know if it breaks or bugs") if (self.verbose) else 0

		# Preformatting the output
		these_X = []
		these_Y = []
		these_ID = []
		these_TC = []
		this_ID = 0
		these_dist = []

		freeST = strikethrough("free")

		# Looping through the lines
		for this_line in self.shpdf["geometry"].values:
			print("I want to break " + freeST + " line #%s" %(this_ID)) if self.verbose else 0
			# Reinitialising the distance
			this_dist = 0
			# Reiterating the interpolation while we still are on the line
			## Background about geopandas/shapely: 
			## line.length is the total length of that line;
			## interpolate returns a point at a certain distance from the origin of the line
			## xy returns weird coordinates from a point object
			while(this_dist < this_line.length):
				# Getting the point
				tC = this_line.interpolate(this_dist)
				# Saving the point
				these_TC.append(tC)
				# Extracting raw x and y
				these_X.append(tC.xy[0][0])
				these_Y.append(tC.xy[1][0])
				these_ID.append(this_ID)
				these_dist.append(this_dist)
				# Incrementing the interval
				this_dist += interval

			# Line is done yo, incrementing the ID
			this_ID += 1
		# Formatting output csv
		odf = pd.DataFrame({"ID":these_ID, "X": these_X, "Y": these_Y, "dist_along_line":these_dist})
		
		if(save_to_shapefile):
			print("Saving to shapefile ... %s"%(self.path + self.prefix + "_line2points.shp")) if self.verbose else 0
			# Formatting shapefile output
			ogdf = odf.copy()
			ogdf["coordinates"] = these_TC
			ogdf = gpd.GeoDataFrame(ogdf, geometry = "coordinates")
			ogdf.crs = self.shpdf.crs
			ogdf.to_file(self.path + self.prefix + "_line2points.shp")
		if(save_to_csv):
			print("Saving to csv ... %s"%(self.path + self.prefix + "_line2points.csv")) if self.verbose else 0
			odf.to_csv(self.path + self.prefix + "_line2points.csv", index = False)

		print("Done with breaking your lines.") if self.verbose else 0

		return odf
	


	def test_rectangular_window(self, distance_along_line = 200, half_width = 600, sample_step = 20, line2points = True, point = 25):
		"""
			Test function to visualise a rectangular window.
			Description to make later
		"""
		from scipy import stats

		if(line2points):
			points = self.line_to_points(interval = sample_step)
		print(points)

		# number of points to get on each side of the reference node on the line
		half_n_point_reg = round(distance_along_line / sample_step / 2)

		# number of points to get on each side of the reference node on the line
		half_n_point_side = round(half_width / sample_step / 2)

		# test function, so I am just taking the first line

		# First step is to get the base points
		Xes = points["X"].values[point - half_n_point_reg : point+ half_n_point_reg];Yes = points["Y"].values[point - half_n_point_reg: point+ half_n_point_reg]
		txmin = np.min(Xes);txmax = np.max(Xes)

		# The linear regression estimates the best m x + b for that part
		this_m, this_b, r_value, p_value, std_err = stats.linregress(Xes,Yes)

		alpha = np.arctan(this_m)

		# Now getting the resampled points
		X_A = txmin
		Y_A = this_m * X_A + this_b

		dx = sample_step * np.cos(alpha)
		dy = sample_step * np.sin(alpha)

		# sampling the new baseline
		est_Xes = [X_A]
		est_Yes = [Y_A]
		this_dist = 0
		while(this_dist < distance_along_line):
			# Calculating the new coordinates
			X_B = X_A + dx
			Y_B = Y_A + dy
			# Incrementing the cat stuffs
			this_dist += sample_step
			# Saving and moving on
			est_Xes.append(X_B)
			est_Yes.append(Y_B)
			X_A = X_B
			Y_A = Y_B

		# Alright, Now I do have the base line, I just need to find a way to get the orthogonal stuffs innit?
		list_of_orthogonal_X = []
		list_of_orthogonal_Y = []
		barking_m = -1/this_m
		barking_alpha = np.arctan(barking_m)
		barking_dx = sample_step * np.cos(barking_alpha)
		barking_dy = sample_step * np.sin(barking_alpha)
		for i in range(len(est_Xes)):
			# Just getting the bloody coordinates of the assessed point
			tx = est_Xes[i]
			ty = est_Yes[i]

			this_dist = 0
			this_barking_Xes = [tx]
			this_barking_Yes = [ty]
			increment = 1
			while(this_dist<half_width):
				this_barking_Xes.append(tx + increment * barking_dx)
				this_barking_Xes.append(tx - increment * barking_dx)
				this_barking_Yes.append(ty + increment * barking_dy)
				this_barking_Yes.append(ty - increment * barking_dy)
				this_dist += sample_step
				increment += 1
			list_of_orthogonal_X.append(this_barking_Xes)
			list_of_orthogonal_Y.append(this_barking_Yes)

		# TRUE OUTPUT
		return list_of_orthogonal_X, list_of_orthogonal_Y
		# FAKE OUTPUT FOR TESTING PURPOSES
		# return est_Xes, est_Yes

	def rectangular_window_from_line2point_df(self, df, this_iloc, sample_step, distance_along_line = 200, half_width = 600):
		"""
			Test function to visualise a rectangular window.
			Description to make later
		"""
		from scipy import stats

		# number of points to get on each side of the reference node on the line
		half_n_point_reg = round(distance_along_line / sample_step / 2)

		# number of points to get on each side of the reference node on the line
		half_n_point_side = round(half_width / sample_step / 2)

		Xes = []
		Yes = []

		# I am ignoring that point at the moment !
		if(this_iloc<half_n_point_reg or this_iloc> df.shape[0]-half_n_point_reg):
			return [-9999], [-9999]

		# First step is to get the base points
		Xes = df["X"].values[this_iloc - half_n_point_reg : this_iloc+ half_n_point_reg];Yes = df["Y"].values[this_iloc - half_n_point_reg: this_iloc+ half_n_point_reg]
		txmin = np.min(Xes);txmax = np.max(Xes)

		# The linear regression estimates the best m x + b for that part
		this_m, this_b, r_value, p_value, std_err = stats.linregress(Xes,Yes)

		alpha = np.arctan(this_m)

		# Now getting the resampled points
		X_A = txmin
		Y_A = this_m * X_A + this_b

		dx = sample_step * np.cos(alpha)
		dy = sample_step * np.sin(alpha)

		# sampling the new baseline
		est_Xes = [X_A]
		est_Yes = [Y_A]
		this_dist = 0
		while(this_dist < distance_along_line):
			# Calculating the new coordinates
			X_B = X_A + dx
			Y_B = Y_A + dy
			# Incrementing the cat stuffs
			this_dist += sample_step
			# Saving and moving on
			est_Xes.append(X_B)
			est_Yes.append(Y_B)
			X_A = X_B
			Y_A = Y_B

		# Alright, Now I do have the base line, I just need to find a way to get the orthogonal stuffs innit?
		list_of_orthogonal_X = []
		list_of_orthogonal_Y = []
		barking_m = -1/this_m
		barking_alpha = np.arctan(barking_m)
		barking_dx = sample_step * np.cos(barking_alpha)
		barking_dy = sample_step * np.sin(barking_alpha)
		for i in range(len(est_Xes)):
			# Just getting the bloody coordinates of the assessed point
			tx = est_Xes[i]
			ty = est_Yes[i]

			this_dist = 0
			this_barking_Xes = [tx]
			this_barking_Yes = [ty]
			increment = 1
			while(this_dist<half_width):
				this_barking_Xes.append(tx + increment * barking_dx)
				this_barking_Xes.append(tx - increment * barking_dx)
				this_barking_Yes.append(ty + increment * barking_dy)
				this_barking_Yes.append(ty - increment * barking_dy)
				this_dist += sample_step
				increment += 1
			list_of_orthogonal_X.append(this_barking_Xes)
			list_of_orthogonal_Y.append(this_barking_Yes)

		# TRUE OUTPUT
		return list_of_orthogonal_X, list_of_orthogonal_Y



		













######## Random small functions

def strikethrough(text):
    result = ''
    for c in text:
        result = result + c + '\u0336'
    return result