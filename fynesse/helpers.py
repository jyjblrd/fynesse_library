from .config import *

from . import access

import math

def km_to_lat_long(distance_km, latitude):
  # Radius of the Earth in kilometers
  earth_radius_km = 6371.0

  # Calculate the angular difference in latitude (in degrees)
  lat_diff = (distance_km / earth_radius_km) * (180.0 / math.pi)

  # Calculate the angular difference in longitude (in degrees)
  # The correction factor (cosine of latitude) accounts for the decreasing longitude spacing with increasing latitude
  lon_diff = (distance_km / (earth_radius_km * math.cos(math.radians(latitude)))) * (180.0 / math.pi)

  return lat_diff, lon_diff

def gen_bounding_box(latitude, longitude, size):
  box_height, box_width = km_to_lat_long(size, latitude)

  north = latitude + box_height/2
  south = latitude - box_height/2
  west = longitude - box_width/2
  east = longitude + box_width/2

  return north, south, east, west

def normalize(array):
  return (array-array.min())/(array.max()-array.min())

def remove_outliers_iqr(data, multiplier=1.5):
  """
  Remove outliers from a 1-dimensional NumPy array using the interquartile range (IQR) method.

  Parameters:
  - data (numpy.ndarray): 1-dimensional array containing the data.
  - multiplier (float): Multiplier to control the sensitivity of outlier detection.

  Returns:
  - numpy.ndarray: Array without outliers.
  """
  # Calculate the first and third quartiles
  Q1 = np.percentile(data, 25)
  Q3 = np.percentile(data, 75)

  # Calculate the IQR (Interquartile Range)
  IQR = Q3 - Q1

  # Define the lower and upper bounds for outlier detection
  lower_bound = Q1 - multiplier * IQR
  upper_bound = Q3 + multiplier * IQR

  # Remove outliers
  filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

  return filtered_data