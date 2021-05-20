"""
These tools are probably there because I needed very quick test of relatively computationally expensive shits that will get ported to cpp when I'll have time to actually compile stuffs.
best,
Boris
"""

import numpy as np
import numba as nb
import math
M_PI = np.pi

# @nb.jit(nopython = True)
# def thin_perimeter_for_salesman(X,Y,distance_thinning):

#   new_index = np.zeros(X.shape[0])


#   return new_index

@nb.jit(nopython = True)
def travelling_salesman_algortihm(X,Y):

  is_visited = np.zeros(X.shape[0])
  temp_distance = np.zeros(X.shape[0])
  new_index = np.zeros(X.shape[0], dtype = np.int32)

  new_index[0] = 0
  is_visited[0] = 1
  that_index = 0

  for i in range(1,new_index.shape[0]):

    that_x = X[that_index]
    that_y = Y[that_index]

    min_distance = 1e32
    min_id = -9999

    #compute distance
    temp_distance[that_index] = min_distance
    for j in range(new_index.shape[0]):
      if(j!= that_index):
        temp_distance[j] = math.sqrt(abs(X[j] - that_x)**2 + abs(Y[j] - that_y) **2)
        if(temp_distance[j] < min_distance and is_visited[j] == 0):
          min_distance = temp_distance[j]
          min_id = j

    new_index[i] = min_id
    that_index = min_id
    is_visited[that_index] = 1

  return new_index

@nb.jit(nopython = True)
def remove_outliers_in_drainage_divide(Dx, Dy, threshold):

  indices = np.full(Dx.shape[0], True)

  is_outlier = False
  last = 0

  for i in range(Dx.shape[0]):
    if(Dx[i]>threshold and Dy[i]>threshold):
      is_outlier = True

    if(is_outlier):
      indices[i] = False

  return indices



@nb.jit(nopython=False, parallel = True)
def average_from_grid_n_stuff_maybe_adaptative_and_all(X,Y,Z,window):
	"""
	Take X,Y,Z and a window size. Return an array of resampled stuffs from that window size.
	Not sure how it exacty works
	B.G.
	"""

	# First getting the minimum and maximum
	Xminimum = np.min(X)
	Yminimum = np.min(Y)
	Xmaximum = np.max(X)
	Ymaximum = np.max(Y)
	# print(Xminimum)
	# print(Yminimum)
	# print(Xmaximum)
	# print(Ymaximum)

	n_cols = np.int32(round((Xmaximum-Xminimum)/window))
	n_rows = np.int32(round((Ymaximum-Yminimum)/window))
	
	avevarray = np.zeros((n_rows,n_cols))
	n = np.zeros((n_rows,n_cols))

	X_coord = np.zeros(n_cols)
	Y_coord = np.zeros(n_rows)

	dat_X = Xminimum
	for i in range(n_cols):
		# print(dat_X)
		X_coord[i] = dat_X + (window)/2
		dat_X += window

	dat_Y = Yminimum
	for i in range(n_rows):
		Y_coord[i] = dat_Y + (window)/2
		dat_Y += window
	# Inverting Y coordinate
	Y_coord = Y_coord[::-1]

	helfwind = window/2
	for i in nb.prange(n_rows):
		for j in range(n_cols):
			# print("BITE")
			if(Z[((X<X_coord[j]+helfwind) & (X>=X_coord[j]-helfwind) & (Y < Y_coord[i]+helfwind) & (Y >= Y_coord[i]-helfwind) ) ].shape[0]>0):
				avevarray[i,j] = np.median(Z[((X<X_coord[j]+helfwind) & (X>=X_coord[j]-helfwind) & (Y < Y_coord[i]+helfwind) & (Y >= Y_coord[i]-helfwind) ) ] )

	return avevarray, X_coord, Y_coord



def numba_hillshading(arr,res,azimuth = 315,angle_altitude = 45,z_factor = 1., parallel = True, ignore_edge = False):
  """
  Numba mnplementation of the hillshading algorithm present in LSDTT c++ algorithmath.
  This is basically just to train myself before optimization/parrallelization with numba
  BG, from DAV-SWDG-SMM codes in LSDTopoTools
  """

  HSarray = np.empty(arr.shape, dtype = np.float32)
  HSarray.fill(-9999)

  zenith_rad = (90 - angle_altitude) * M_PI / 180.0
  azimuth_math = 360-azimuth + 90

  if (azimuth_math >= 360.0):
    azimuth_math = azimuth_math - 360
  azimuth_rad = azimuth_math * M_PI  / 180.0

  slope_rad = 0
  aspect_rad = 0
  dzdx = 0
  dzdy = 0
  i = 0
  j = 0
  nrows = arr.shape[0] -1
  ncols = arr.shape[1] -1

  if(ignore_edge):
    arr[:,0] = -9999
    arr[:,-1] = -9999

    arr[0,:] = -9999
    arr[-1,:] = -9999
    

  # print("I am hillshading")

  if(not parallel):
    return _hillshading(arr,res,ncols,nrows,HSarray,zenith_rad,azimuth_rad, z_factor)
  elif(parallel):
    return _parallel_hillshading(arr,res,ncols,nrows,HSarray,zenith_rad,azimuth_rad, z_factor)

@nb.jit(nopython = True,cache = True)
def _hillshading(arr,res,ncols,nrows,HSarray,zenith_rad,azimuth_rad, z_factor):
  """
  Slow and basic imnplementation of the hillshading algorithm present in LSDTT c++ algorithmath.
  This is basically just to train myself before optimization/parrallelization with numba
  BG, from DAV-SWDG-SMM codes
  """

  for i in range(nrows):
    for j in range(ncols):

      if arr[i, j] != -9999:

        dzdx = (((arr[i, j+1] + 2*arr[i+1, j] + arr[i+1, j+1]) -
                (arr[i-1, j-1] + 2*arr[i-1, j] + arr[i-1, j+1]))
                / (8 * res))
        dzdy = (((arr[i-1, j+1] + 2*arr[i, j+1] + arr[i+1, j+1]) -
                (arr[i-1, j-1] + 2*arr[i, j-1] + arr[i+1, j-1]))
                / (8 * res))

        slope_rad = np.arctan(z_factor * np.sqrt((dzdx * dzdx) + (dzdy * dzdy)))

        if (dzdx != 0):
          aspect_rad = np.arctan2(dzdy, (dzdx*-1))
          if (aspect_rad < 0):
            aspect_rad = 2 * M_PI + aspect_rad

        else:
          if (dzdy > 0):
            aspect_rad = M_PI / 2
          elif (dzdy < 0):
            aspect_rad = 2 * M_PI - M_PI / 2
          else:
            aspect_rad = aspect_rad

        HSarray[i, j] = 255.0 * ((np.cos(zenith_rad) * np.cos(slope_rad)) +
                        (np.sin(zenith_rad) * np.sin(slope_rad) *
                        np.cos(azimuth_rad - aspect_rad)))
        # Necessary?
        if (HSarray[i, j] < 0):
          HSarray[i, j] = 0
  
  return HSarray 

@nb.jit(nopython = True, parallel = True)
def _parallel_hillshading(arr,res,ncols,nrows,HSarray,zenith_rad,azimuth_rad, z_factor):
  """
  Slow and basic imnplementation of the hillshading algorithm present in LSDTT c++ algorithmath.
  This is basically just to train myself before optimization/parrallelization with numba
  BG, from DAV-SWDG-SMM codes
  """

  for i in nb.prange(nrows):
    for j in range(ncols):

      if arr[i, j] != -9999:

        dzdx = (((arr[i, j+1] + 2*arr[i+1, j] + arr[i+1, j+1]) -
                (arr[i-1, j-1] + 2*arr[i-1, j] + arr[i-1, j+1]))
                / (8 * res))
        dzdy = (((arr[i-1, j+1] + 2*arr[i, j+1] + arr[i+1, j+1]) -
                (arr[i-1, j-1] + 2*arr[i, j-1] + arr[i+1, j-1]))
                / (8 * res))

        slope_rad = np.arctan(z_factor * np.sqrt((dzdx * dzdx) + (dzdy * dzdy)))

        if (dzdx != 0):
          aspect_rad = np.arctan2(dzdy, (dzdx*-1))
          if (aspect_rad < 0):
            aspect_rad = 2 * M_PI + aspect_rad

        else:
          if (dzdy > 0):
            aspect_rad = M_PI / 2
          elif (dzdy < 0):
            aspect_rad = 2 * M_PI - M_PI / 2
          else:
            aspect_rad = aspect_rad

        HSarray[i, j] = 255.0 * ((np.cos(zenith_rad) * np.cos(slope_rad)) +
                        (np.sin(zenith_rad) * np.sin(slope_rad) *
                        np.cos(azimuth_rad - aspect_rad)))
        # Necessary?
        if (HSarray[i, j] < 0):
          HSarray[i, j] = 0
  
  return HSarray