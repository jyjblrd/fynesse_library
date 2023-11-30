from .config import *

import pymysql
from pymysql.constants import CLIENT
import osmnx as ox
from geopy.geocoders import Nominatim
import urllib.request
import os
import zipfile
from .helpers import gen_bounding_box

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def create_database_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database,
                               client_flag=CLIENT.MULTI_STATEMENTS,
                               autocommit=True
                               )
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def get_pois(latitude, longitude, size, tags):
  """
  Query OSM api to get pois with certain tags in a certain area
  """

  north, south, east, west = gen_bounding_box(latitude, longitude, size)

  return ox.features_from_bbox(north, south, east, west, tags)

def get_location_info(lat, lon):
    """
    Convert coodinates into a city name, postcode and country
    """

    # Reverse geocoding using Nominatim
    geolocator = Nominatim(user_agent="my_geocoder")
    location = geolocator.reverse((lat, lon), language='en')

    # Extract relevant information
    address = location.raw.get('address', {})
    city = address.get('city', '')
    country = address.get('country', '')
    postcode = address.get('postcode', '')

    return city, country, postcode