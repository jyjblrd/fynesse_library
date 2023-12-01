from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import geopandas
import osmnx as ox
import math
from shapely.ops import nearest_points
from shapely.geometry import Point
from functools import partial
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.neighbors import BallTree
import geopy.distance
from geopy.geocoders import Nominatim
from .helpers import *
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-muted')


"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

#############################
## Load data from database ##
#############################

def get_db_data(cur, north, south, east, west, start_date, end_date):
  """
  Get data from the database within the given bounding box and dates
  """

  cur.execute("""
    SELECT
      `pp_data`.`price`,
      `pp_data`.`date_of_transfer`,
      `pp_data`.`postcode`,
      `pp_data`.`property_type`,
      `pp_data`.`new_build_flag`,
      `pp_data`.`tenure_type`,
      `pp_data`.`locality`,
      `pp_data`.`town_city`,
      `pp_data`.`district`,
      `pp_data`.`county`,
      `postcode_data`.`country`,
      `postcode_data`.`latitude`,
      `postcode_data`.`longitude`
    FROM `pp_data`
    LEFT OUTER JOIN `postcode_data`
    ON `pp_data`.`postcode` = `postcode_data`.`postcode`
    WHERE
      `postcode_data`.`status` = 'live' AND
      `postcode_data`.`latitude` < %s AND
      `postcode_data`.`latitude` > %s AND
      `postcode_data`.`longitude` < %s AND
      `postcode_data`.`longitude` > %s AND
      `pp_data`.`date_of_transfer` > %s AND
      `pp_data`.`date_of_transfer` < %s;
  """, (north, south, east, west, start_date, end_date))

  data = cur.fetchall()

  return data

def convert_db_data_to_gdf(data):
  """
  Convert the raw data from the database into a geometry dataframe
  with proper columns names, etc.
  """

  df = pd.DataFrame(data, columns=["price", "date_of_transfer", "postcode", "property_type", "new_build_flag", "tenure_type", "locality", "town_city", "district", "county", "country", "latitude", "longitude"])
  
  gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(x=df.longitude, y=df.latitude))

  gdf["date_of_transfer_unix_ns"] = pd.to_datetime(gdf["date_of_transfer"]).astype(int)

  property_type_dummies = pd.get_dummies(df["property_type"], prefix="property_type")
  gdf = pd.concat([gdf, property_type_dummies], axis='columns')
  
  if 'property_type_F' not in gdf:
    gdf['property_type_F'] = 0
  if 'property_type_S' not in gdf:
    gdf['property_type_S'] = 0
  if 'property_type_D' not in gdf:
    gdf['property_type_D'] = 0
  if 'property_type_T' not in gdf:
    gdf['property_type_T'] = 0
  if 'property_type_O' not in gdf:
    gdf['property_type_O'] = 0

  return gdf

def plot_all_parameters(gdf):
  """
  Plot all the parameters used to train our linear model
  """
  fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(20, 15))

  sns.regplot(x="date_of_transfer_unix_ns", y="price", data=gdf, scatter_kws={"alpha": 0.2}, line_kws={"color": "red"}, ax=ax1)
  gdf.plot.scatter("new_build_flag", "price", ax=ax2)
  gdf.plot.scatter("property_type", "price", ax=ax3)
  gdf.plot.scatter("tenure_type", "price", ax=ax4)
  sns.regplot(x="dist_to_city_center", y="price", data=gdf, scatter_kws={"alpha": 0.2}, line_kws={"color": "red"}, ax=ax5)
  sns.regplot(x="dist_to_public_transport", y="price", data=gdf, scatter_kws={"alpha": 0.2}, line_kws={"color": "red"}, ax=ax6)
  sns.regplot(x="dist_to_shop", y="price", data=gdf, scatter_kws={"alpha": 0.2}, line_kws={"color": "red"}, ax=ax7)
  sns.regplot(x="dist_to_school", y="price", data=gdf, scatter_kws={"alpha": 0.2}, line_kws={"color": "red"}, ax=ax8)
  sns.regplot(x="dist_to_industrial", y="price", data=gdf, scatter_kws={"alpha": 0.2}, line_kws={"color": "red"}, ax=ax9)
  sns.regplot(x="dist_to_recreation_ground", y="price", data=gdf, scatter_kws={"alpha": 0.2}, line_kws={"color": "red"}, ax=ax10)
  sns.regplot(x="dist_to_nature", y="price", data=gdf, scatter_kws={"alpha": 0.2}, line_kws={"color": "red"}, ax=ax11)
  sns.regplot(x="avg_neighbour_price", y="price", data=gdf, scatter_kws={"alpha": 0.2}, line_kws={"color": "red"}, ax=ax12)


#####################
## OSM POI helpers ##
#####################

def plot_pois(pois, graph, ax, north, south, east, west, title):
  """
  Plot OSM pois on a map
  """

  # Plot street edges
  _, edges = ox.graph_to_gdfs(graph)
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

  ax.set_xlim([west, east])
  ax.set_ylim([south, north])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")
  ax.set_title(title)

  # Plot tourist places
  pois.plot(ax=ax, alpha=1, markersize=50)
  plt.tight_layout()

def pois_to_coords(pois):
  """
  Convert OSM pois into coordinates
  """

  return pois["geometry"].reset_index(drop=True).unary_union

def dist_to_poi(poi_coords, property_row):
  """
  Given a property, find the distance to the closest poi
  """
  if poi_coords is None:
    return 0.0
  
  nearest_point = nearest_points(Point(property_row["longitude"], property_row["latitude"]), poi_coords)[1]

  return geopy.distance.distance((float(nearest_point.y), float(nearest_point.x)), (float(property_row["latitude"]), float(property_row["longitude"]))).km

def plot_landuse(pois, graph, ax):
    """
    Plot land use categories in the selected area
    """

    filtered_pois = pois[pois["landuse"].notna()]
    top_landuses = filtered_pois[filtered_pois["landuse"].isin(filtered_pois["landuse"].value_counts().nlargest(10).index)]
    landuse_palette = sns.color_palette("Set3", n_colors=len(top_landuses["landuse"].unique()))

    # Plot landuse
    for idx, landuse_category in enumerate(top_landuses["landuse"].unique()):
        subset = filtered_pois[filtered_pois["landuse"] == landuse_category]
        subset.plot(ax=ax, color=landuse_palette[idx])
    
    # Plot street edges
    _, edges = ox.graph_to_gdfs(graph)
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.legend(title="Land Use", loc="upper right", bbox_to_anchor=(1.3, 1), handles=[mpatches.Patch(color=landuse_palette[i], label=landuse_category) for i, landuse_category in enumerate(top_landuses["landuse"].unique())])

def get_nearest_points(src_point, candidate_gdf, k_neighbors=1):
    """
    Find nearest neighbors for all source points from a set of candidate points
    """

    src_point = np.array((src_point.x * np.pi / 180, src_point.y * np.pi / 180)).reshape(1, -1)
    candidates = np.array(candidate_gdf["geometry"].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_point, k=k_neighbors)

    # Return indices and distances
    return candidate_gdf.iloc[indices[0]]

def add_dist_to_city_center(gdf, size):
  city_center_cache = {}
  def dist_to_city_center(row):
    city_name = f'{row["town_city"]}, {row["country"]}'

    try:
      if city_name in city_center_cache:
        city_center = city_center_cache[city_name]
      else:
          city_center = ox.geocode_to_gdf(city_name).centroid
          city_center_cache[city_name] = city_center
      return geopy.distance.distance((float(city_center.y), float(city_center.x)), (float(row["latitude"]), float(row["longitude"]))).km
    except:
      return float(size/2)

  gdf["dist_to_city_center"] = gdf.apply(dist_to_city_center, axis=1)

  return gdf


#########################
## Neighbour functions ##
#########################

def generate_neighbour_metrics_single_property(property, city_gdf, neighbour_radius):
  # only look at properties within neighbour_radius km
  north, south, east, west = gen_bounding_box(float(property["latitude"]), float(property["longitude"]), neighbour_radius*2)

  # Get neighbours within bounding box
  neighbours = city_gdf[(city_gdf["latitude"] < north) & (city_gdf["latitude"] > south) & (city_gdf["longitude"] < east) & (city_gdf["longitude"] > west)]

  # remove current property from the sample (obviously)
  if property.name in neighbours.index:
    neighbours = neighbours.drop(index=property.name)

  price_difference_pct = np.abs(neighbours["price"] - property["price"]) / property["price"] if property["price"] is not None else None
  distance_to_neighbour = neighbours["geometry"].distance(property["geometry"])
  property_type_same = (property["property_type"] == neighbours["property_type"]).astype(float)
  date_delta = np.abs(property["date_of_transfer_unix_ns"] - neighbours["date_of_transfer_unix_ns"])

  neighbour_metrics = pd.DataFrame({
      "price_difference_pct": price_difference_pct,
      "distance_to_neighbour": distance_to_neighbour,
      "property_type_same": property_type_same,
      "date_delta": date_delta,
  })

  return neighbour_metrics, neighbours

def generate_neighbour_training_df(gdf, neighbour_radius):
  """
  Generate a dataframe containing data used to train a linear model
  to select the most representative neighbours, ie. neighbours that
  are most likely to have similar prices.
  """

  # dont bother include street name cus postcode is generally more granular
  neighbour_training_df = pd.DataFrame(columns=["price_difference_pct", "distance_to_neighbour", "property_type_same", "new_build_flag_same", "tenure_type_same"])

  for index, row in gdf.iterrows():
    neighbour_metrics, neighbours = generate_neighbour_metrics_single_property(row, gdf, neighbour_radius)
    if len(neighbour_metrics) != 0:
      neighbour_training_df = pd.concat((neighbour_training_df, neighbour_metrics))

  return neighbour_training_df

def plot_neighbour_training_df(neighbour_training_df):
  """
  Plot the data in the neighbour_training_df
  """

  fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 5))

  neighbour_training_df.plot.scatter("distance_to_neighbour", "price_difference_pct", ax=ax1, alpha=0.01)
  ax2.scatter(neighbour_training_df["property_type_same"].astype(int), neighbour_training_df["price_difference_pct"], alpha=0.003)
  ax2.set_xlabel("property_type_same")
  ax2.set_ylabel("price_difference_pct")
  neighbour_training_df.plot.scatter("date_delta", "price_difference_pct", ax=ax3, alpha=0.01)

def remove_null_nieghbour_prices(gdf):
  """
  Replace null neighbour prices with the mean price in the area.
  This is used when a property has no neighbours.
  """

  print(f"number of null neighbour prices: {gdf['avg_neighbour_price'].isna().sum()}")
  # because there may be some null values
  gdf["avg_neighbour_price"] = gdf["avg_neighbour_price"].fillna(gdf["avg_neighbour_price"].mean())

  return gdf

def plot_neighbour_model_error(predicted, actual):
  """
  Plot the errors of the neighbour model
  """

  fig, ((ax1)) = plt.subplots(1, 1, figsize=(5, 5))

  neighbour_errors = (actual - predicted) * 100
  neighbour_errors.name = "neighbour_errors"

  colors = np.zeros(neighbour_errors.size)

  bins = 1000
  mean_error = neighbour_errors.mean()
  std_dev = neighbour_errors.std()

  for i in range(-3, 4):
      ax1.axvline(mean_error + i * std_dev, color='lightblue', linewidth=1)

  ax1.hist(neighbour_errors, bins=bins, alpha=0.7)


  ax1.set_xlabel('neighbour_Errors')
  ax1.set_ylabel('Frequency')
  ax1.set_title('neighbour_Errors w/ Standard Deviations')
  ax1.set_xlim((mean_error + -3 * std_dev, mean_error + 3 * std_dev))

  print(pd.DataFrame(neighbour_errors).describe())


###########################
## Price model functions ##
###########################

def plot_model_errors(predicted, actual):
  errors = actual - predicted
  errors.name = "errors"

  colors = np.zeros(errors.size)

  bins = 30
  mean_error = errors.mean()
  std_dev = errors.std()

  fig, ((ax1)) = plt.subplots(1, 1)

  for i in range(-3, 4):
      ax1.axvline(mean_error + i * std_dev, color='lightblue', linewidth=1)

  hist, edges, _ = ax1.hist(errors, bins=bins, alpha=0.7)

  ax1.set_xlabel('Errors')
  ax1.set_ylabel('Frequency')
  ax1.set_title('Errors w/ Standard Deviations')
  ax1.set_xlim((mean_error + -3 * std_dev, mean_error + 3 * std_dev))
  plt.show()

  print(pd.DataFrame(errors).describe())