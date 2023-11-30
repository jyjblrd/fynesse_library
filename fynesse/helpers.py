from .config import *

from . import access

import math
import numpy as np

def km_to_lat_long(distance_km, latitude):
  """
  Convert a box of size distance_km into a box defined by coordinates
  """

  # Radius of the Earth in kilometers
  earth_radius_km = 6371.0

  # Calculate the angular difference in latitude (in degrees)
  lat_diff = (distance_km / earth_radius_km) * (180.0 / math.pi)

  # Calculate the angular difference in longitude (in degrees)
  # The correction factor (cosine of latitude) accounts for the decreasing longitude spacing with increasing latitude
  lon_diff = (distance_km / (earth_radius_km * math.cos(math.radians(latitude)))) * (180.0 / math.pi)

  return lat_diff, lon_diff

def gen_bounding_box(latitude, longitude, size):
  """
  Given coordinates and a size, generate a bounding box
  """

  box_height, box_width = km_to_lat_long(size, latitude)

  north = latitude + box_height/2
  south = latitude - box_height/2
  west = longitude - box_width/2
  east = longitude + box_width/2

  return north, south, east, west

def normalize(array):
  """
  Normalize an array from 0-1
  """

  return (array-array.min())/(array.max()-array.min())

def remove_outliers_iqr(data, multiplier=1.5):
  """
  Remove outliers using the iqr method
  """

  Q1 = np.percentile(data, 25)
  Q3 = np.percentile(data, 75)

  IQR = Q3 - Q1

  lower_bound = Q1 - multiplier * IQR
  upper_bound = Q3 + multiplier * IQR

  # Remove outliers
  filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

  return filtered_data