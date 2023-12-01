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

#####################################
## Create Databse Schema Functions ##
#####################################

def create_pp_data_table(cur):
  # Create pp_data table
  cur.execute("""
    DROP TABLE IF EXISTS `pp_data`;
    CREATE TABLE IF NOT EXISTS `pp_data` (
      `transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,
      `price` int(10) unsigned NOT NULL,
      `date_of_transfer` date NOT NULL,
      `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
      `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
      `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
      `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
      `primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
      `secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
      `street` tinytext COLLATE utf8_bin NOT NULL,
      `locality` tinytext COLLATE utf8_bin NOT NULL,
      `town_city` tinytext COLLATE utf8_bin NOT NULL,
      `district` tinytext COLLATE utf8_bin NOT NULL,
      `county` tinytext COLLATE utf8_bin NOT NULL,
      `ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,
      `record_status` varchar(2) COLLATE utf8_bin NOT NULL,
      `db_id` bigint(20) unsigned NOT NULL
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;
  """)

  # Set pp_data's index
  cur.execute("""
    ALTER TABLE `pp_data`
    ADD PRIMARY KEY (`db_id`);

    ALTER TABLE `pp_data`
    MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1;
  """)

def create_postcode_data_table(cur):
  # Create postcode_data table
  cur.execute("""
    DROP TABLE IF EXISTS `postcode_data`;
    CREATE TABLE IF NOT EXISTS `postcode_data` (
      `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
      `status` enum('live','terminated') NOT NULL,
      `usertype` enum('small', 'large') NOT NULL,
      `easting` int unsigned,
      `northing` int unsigned,
      `positional_quality_indicator` int NOT NULL,
      `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
      `latitude` decimal(11,8) NOT NULL,
      `longitude` decimal(10,8) NOT NULL,
      `postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,
      `postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,
      `postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,
      `postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,
      `postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,
      `postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,
      `outcode` varchar(4) COLLATE utf8_bin NOT NULL,
      `incode` varchar(3)  COLLATE utf8_bin NOT NULL,
      `db_id` bigint(20) unsigned NOT NULL
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
  """)

  # Set postcode_data's index
  cur.execute("""
    ALTER TABLE `postcode_data`
    ADD PRIMARY KEY (`db_id`);

    ALTER TABLE `postcode_data`
    MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;
  """)

def create_property_prices_database(cur):
  # Create property_prices database
  cur.execute("""
    CREATE DATABASE IF NOT EXISTS `property_prices` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;
    USE `property_prices`;
  """)