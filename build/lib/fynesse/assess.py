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


"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


# def data():
#     """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
#     df = access.data()
#     raise NotImplementedError

# def query(data):
#     """Request user input for some aspect of the data."""
#     raise NotImplementedError

# def view(data):
#     """Provide a view of the data that allows the user to verify some aspect of its quality."""
#     raise NotImplementedError

# def labelled(data):
#     """Provide a labelled set of data ready for supervised learning."""
#     raise NotImplementedError



#############################
## Load data from database ##
#############################

def get_db_data(cur, north, south, east, west, start_date, end_date):
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
  sns.lmplot(x="date_of_transfer_unix_ns", y="price", data=gdf, scatter_kws={"alpha": 0.01}, line_kws={"color": "red"})
  gdf.plot.scatter("new_build_flag", "price")
  gdf.plot.scatter("property_type", "price")
  gdf.plot.scatter("tenure_type", "price")
  sns.lmplot(x="dist_to_city_center", y="price", data=gdf, scatter_kws={"alpha": 0.01}, line_kws={"color": "red"})
  sns.lmplot(x="dist_to_public_transport", y="price", data=gdf, scatter_kws={"alpha": 0.01}, line_kws={"color": "red"})
  sns.lmplot(x="dist_to_shop", y="price", data=gdf, scatter_kws={"alpha": 0.01}, line_kws={"color": "red"})
  sns.lmplot(x="dist_to_school", y="price", data=gdf, scatter_kws={"alpha": 0.01}, line_kws={"color": "red"})
  sns.lmplot(x="dist_to_industrial", y="price", data=gdf, scatter_kws={"alpha": 0.01}, line_kws={"color": "red"})
  sns.lmplot(x="dist_to_recreation_ground", y="price", data=gdf, scatter_kws={"alpha": 0.01}, line_kws={"color": "red"})
  sns.lmplot(x="dist_to_nature", y="price", data=gdf, scatter_kws={"alpha": 0.01}, line_kws={"color": "red"})
  sns.lmplot(x="avg_neighbour_price", y="price", data=gdf, scatter_kws={"alpha": 0.01}, line_kws={"color": "red"})


#####################
## OSM POI helpers ##
#####################

def plot_pois(pois, graph, ax):
  # Plot street edges
  _, edges = ox.graph_to_gdfs(graph)
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

  ax.set_xlim([west, east])
  ax.set_ylim([south, north])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")

  # Plot tourist places
  pois.plot(ax=ax, color="blue", alpha=1, markersize=50)
  plt.tight_layout()

def pois_to_coords(pois):
  return pois["geometry"].reset_index(drop=True).unary_union

def dist_to_poi(poi_coords, row):
  nearest_public_transport = nearest_points(Point(row["longitude"], row["latitude"]), poi_coords)[1]

  return geopy.distance.distance((float(nearest_public_transport.y), float(nearest_public_transport.x)), (float(row["latitude"]), float(row["longitude"]))).km

def plot_landuse(pois, graph, ax):
    filtered_pois = pois[pois["landuse"].notna()]
    top_landuses = filtered_pois[filtered_pois["landuse"].isin(filtered_pois["landuse"].value_counts().nlargest(10).index)]
    landuse_palette = sns.color_palette("Set3", n_colors=len(top_landuses["landuse"].unique()))

    for idx, landuse_category in enumerate(top_landuses["landuse"].unique()):
        subset = filtered_pois[filtered_pois["landuse"] == landuse_category]
        subset.plot(ax=ax, color=landuse_palette[idx])
    
    # Plot street edges
    _, edges = ox.graph_to_gdfs(graph)
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.legend(title="Land Use", loc="upper right", bbox_to_anchor=(1.3, 1), handles=[mpatches.Patch(color=landuse_palette[i], label=landuse_category) for i, landuse_category in enumerate(top_landuses["landuse"].unique())])

def get_nearest_points(src_point, candidate_gdf, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    src_point = np.array((src_point.x * np.pi / 180, src_point.y * np.pi / 180)).reshape(1, -1)
    candidates = np.array(candidate_gdf["geometry"].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_point, k=k_neighbors)

    # Return indices and distances
    return candidate_gdf.iloc[indices[0]]


#########################
## Neighbour functions ##
#########################

def generate_neighbour_training_df(gdf):
  # dont bother include street name cus postcode is generally more granular
  neighbour_training_df = pd.DataFrame(columns=["price_difference_pct", "distance_to_neighbour", "property_type_same", "new_build_flag_same", "tenure_type_same"])

  for index, row in gdf.iterrows():
    print(index)
    # only look at properties within 1km
    north, south, east, west = gen_bounding_box(float(row["latitude"]), float(row["longitude"]), RADIUS*2)

    # Get neighbours within bounding box
    neighbours = gdf[(gdf["latitude"] < north) & (gdf["latitude"] > south) & (gdf["longitude"] < east) & (gdf["longitude"] > west)]

    # remove current property from the sample (obviously)
    neighbours = neighbours.drop(index=row.name)

    price_difference_pct = np.abs(neighbours["price"] - row["price"]) / row["price"]
    distance_to_neighbour = neighbours["geometry"].distance(row["geometry"])
    property_type_same = row["property_type"] == neighbours["property_type"]
    new_build_flag_same = row["new_build_flag"] == neighbours["new_build_flag"]
    tenure_type_same = row["tenure_type"] == neighbours["tenure_type"]

    neighbour_training_df = pd.concat((neighbour_training_df, pd.DataFrame({
        "price_difference_pct": price_difference_pct,
        "distance_to_neighbour": distance_to_neighbour,
        "property_type_same": property_type_same,
        "new_build_flag_same": new_build_flag_same,
        "tenure_type_same": tenure_type_same
    })))

  return neighbour_training_df

def plot_neighbour_training_df(neighbour_training_df):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

  neighbour_training_df.plot.scatter("distance_to_neighbour", "price_difference_pct", ax=ax1, alpha=0.01)
  ax2.scatter(neighbour_training_df["property_type_same"].astype(int), neighbour_training_df["price_difference_pct"], alpha=0.003)

def remove_null_nieghbour_prices(gdf):
  print(f"number of null neighbour prices: {gdf['avg_neighbour_price'].isna().sum()}")
  # because there may be some null values
  gdf["avg_neighbour_price"] = gdf["avg_neighbour_price"].fillna(gdf["avg_neighbour_price"].mean())

  return gdf


