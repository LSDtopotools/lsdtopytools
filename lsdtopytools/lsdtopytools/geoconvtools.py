"""
This file contains very basic but optimised routines for converting coordinates from WGS to UTM and reverse.
Highly adapted from https://github.com/Turbo87/utm, a lightweight utm packaged.
I numbaised it to ingest numpy array/ list of coordinates.
B.G.
"""
import numpy as np
import math
import numba as nb




################################################################################
################################################################################
# This is an optimization from https://github.com/Turbo87/utm
################################################################################
################################################################################


K0 = 0.9996

E = 0.00669438
E2 = E * E
E3 = E2 * E
E_P2 = E / (1.0 - E)

SQRT_E = math.sqrt(1 - E)
_E = (1 - SQRT_E) / (1 + SQRT_E)
_E2 = _E * _E
_E3 = _E2 * _E
_E4 = _E3 * _E
_E5 = _E4 * _E

M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
M3 = (15 * E2 / 256 + 45 * E3 / 1024)
M4 = (35 * E3 / 3072)

P2 = (3. / 2 * _E - 27. / 32 * _E3 + 269. / 512 * _E5)
P3 = (21. / 16 * _E2 - 55. / 32 * _E4)
P4 = (151. / 96 * _E3 - 417. / 128 * _E5)
P5 = (1097. / 512 * _E4)

R = 6378137

ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"


def to_latlon(easting, northing, zone_number, northern=None):
  """
    Main function to convert UTM to latitude-longitude.
    Takes XY and returns latlon basically.
    Arguments:
      easting (float or array): list of easting (X) coordinates
      northing (float or array): list of northing (Y) coordinates
      zone_number: the inputed UTM zone -> http://www.jaworski.ca/utmzones.htm
      northern (Boolean): are you in the northern hemisphere?
    returns:
      Latitude, longitude arrays
    Authors:
      B.G.
    Date:
      At some points in 2018

  """


  if(isinstance(easting,list)):
    easting = np.asarray(easting)
    northing = np.asarray(northing)

  if(isinstance(easting,np.ndarray)):
    return _to_latlon(easting, northing, zone_number,  northern)
  else:
    return _to_latlon([easting], [northing], zone_number,  northern)


@nb.jit(nopython = False)
def _to_latlon(easting, northing, zone_number, northern):

    """Internal function convert an UTM coordinate into Latitude and Longitude
        Parameters
        ----------
        easting: int
            Easting value of UTM coordinate
        northing: int
            Northing value of UTM coordinate
        zone number: int
            Zone Number is represented with global map numbers of an UTM Zone
            Numbers Map. More information see utmzones [1]_
        zone_letter: str
            Zone Letter can be represented as string values. Where UTM Zone
            Designators can be accessed in [1]_
        northern: bool
            You can set True or False to set this parameter. Default is None
       .. _[1]: http://www.jaworski.ca/utmzones.htm
    """

    longitude = []
    latitude = []

    for i in range(easting.shape[0]):

      x = easting[i] - 500000
      y = northing[i]

      if not northern:
          y -= 10000000

      m = y / K0
      mu = m / (R * M1)

      p_rad = (mu +
               P2 * math.sin(2 * mu) +
               P3 * math.sin(4 * mu) +
               P4 * math.sin(6 * mu) +
               P5 * math.sin(8 * mu))

      p_sin = math.sin(p_rad)
      p_sin2 = p_sin * p_sin

      p_cos = math.cos(p_rad)

      p_tan = p_sin / p_cos
      p_tan2 = p_tan * p_tan
      p_tan4 = p_tan2 * p_tan2

      ep_sin = 1 - E * p_sin2
      ep_sin_sqrt = math.sqrt(1 - E * p_sin2)

      n = R / ep_sin_sqrt
      r = (1 - E) / ep_sin

      c = _E * p_cos**2
      c2 = c * c

      d = x / (n * K0)
      d2 = d * d
      d3 = d2 * d
      d4 = d3 * d
      d5 = d4 * d
      d6 = d5 * d

      latitude.append(math.degrees(p_rad - (p_tan / r) *
                        (d2 / 2 -
                         d4 / 24 * (5 + 3 * p_tan2 + 10 * c - 4 * c2 - 9 * E_P2)) +
                         d6 / 720 * (61 + 90 * p_tan2 + 298 * c + 45 * p_tan4 - 252 * E_P2 - 3 * c2)))

      longitude.append(math.degrees(d -
                         d3 / 6 * (1 + 2 * p_tan2 + c) +
                         d5 / 120 * (5 - 2 * c + 28 * p_tan2 - 3 * c2 + 8 * E_P2 + 24 * p_tan4) / p_cos ) + (zone_number - 1) * 6 - 180 + 3)

    return (latitude,longitude)


def from_latlon(latitude, longitude, force_zone_number= -9999):
  """
    Highest level function to convert latitude longitude (WGS84) to UTM.
    Arguments:
      latitude (scalar or list or array): inputted latitude in WGS84
      longitude (scalar or list or array): inputted longitude in WGS84
      force_zone_number (scalar): Zone number will be determined automatically, but you can eventually force one.
    Returns:
      easting,northing (X,Y) coordinate arrays
    Authors:
      B.G. (again from https://github.com/Turbo87/utm)
    Date:
      2018, summer, I think.
  """

  if(isinstance(longitude,list)):
    longitude = np.asarray(longitude)
    latitude = np.asarray(latitude)

  if(isinstance(longitude,np.ndarray)):
    return _from_latlon(latitude, longitude, force_zone_number)
  else:
    return _from_latlon([latitude], [longitude], force_zone_number)



@nb.jit(nopython = True, parallel = True)
def _from_latlon(latitude, longitude, force_zone_number):
    """This function convert Latitude and Longitude to UTM coordinate
        Parameters
        ----------
        latitude: float
            Latitude between 80 deg S and 84 deg N, e.g. (-80.0 to 84.0)
        longitude: float
            Longitude between 180 deg W and 180 deg E, e.g. (-180.0 to 180.0).
        force_zone number: int
            Zone Number is represented with global map numbers of an UTM Zone
            Numbers Map. You may force conversion including one UTM Zone Number.
            More information see utmzones [1]_
       .. _[1]: http://www.jaworski.ca/utmzones.htm
    """
    easting = []
    northing = []

    for i in range(longitude.shape[0]):
      lat_rad = math.radians(latitude[i])
      lat_sin = math.sin(lat_rad)
      lat_cos = math.cos(lat_rad)

      lat_tan = lat_sin / lat_cos
      lat_tan2 = lat_tan * lat_tan
      lat_tan4 = lat_tan2 * lat_tan2

      if(force_zone_number < 0):
        zone_number = latlon_to_zone_number(latitude[i], longitude[i])

      else:
        zone_number = force_zone_number

      lon_rad = math.radians(longitude[i])
      central_lon = (zone_number - 1) * 6 - 180 + 3
      central_lon_rad = math.radians(central_lon)

      n = R / math.sqrt(1 - E * lat_sin**2)
      c = E_P2 * lat_cos**2

      a = lat_cos * (lon_rad - central_lon_rad)
      a2 = a * a
      a3 = a2 * a
      a4 = a3 * a
      a5 = a4 * a
      a6 = a5 * a

      m = R * (M1 * lat_rad -
               M2 * math.sin(2 * lat_rad) +
               M3 * math.sin(4 * lat_rad) -
               M4 * math.sin(6 * lat_rad))

      easting.append(K0 * n * (a +
                          a3 / 6 * (1 - lat_tan2 + c) +
                          a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000)

      northing.append(K0 * (m + n * lat_tan * (a2 / 2 +
                                          a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                          a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2))))

      if latitude[i] < 0:
          northing[-1] += 10000000

    return easting, northing, zone_number


# def latitude_to_zone_letter(latitude):
#     if -80 <= latitude <= 84:
#         return ZONE_LETTERS[int(latitude + 80) >> 3]
#     else:
#         return None

@nb.jit(nopython = True, cache = True)
def latlon_to_zone_number(latitude, longitude):
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude < 9:
            return 31
        elif longitude < 21:
            return 33
        elif longitude < 33:
            return 35
        elif longitude < 42:
            return 37

    return int((longitude + 180) / 6) + 1