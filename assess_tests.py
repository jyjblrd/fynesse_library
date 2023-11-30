#!/usr/bin/env python

import unittest
import fynesse.assess as assess
import fynesse.access as access
import pandas as pd
from shapely.geometry import Point


class AccessTests(unittest.TestCase):

    def test_convert_db_data_to_gdf(self):
        gdf = assess.convert_db_data_to_gdf(pd.DataFrame())

        correct = ['price', 'date_of_transfer', 'postcode', 'property_type',
        'new_build_flag', 'tenure_type', 'locality', 'town_city', 'district',
        'county', 'country', 'latitude', 'longitude', 'geometry',
        'date_of_transfer_unix_ns', 'property_type_F', 'property_type_S',
        'property_type_D', 'property_type_T', 'property_type_O']
        
        assert len(gdf.columns) == len(correct) and all(elem1 == elem2 for elem1, elem2 in zip(gdf.columns, correct))

    def test_plot_all_parameters(self):
        gdf_with_all_columns = assess.convert_db_data_to_gdf(pd.DataFrame())
        gdf_with_all_columns["dist_to_city_center"] = 0
        gdf_with_all_columns["dist_to_public_transport"] = 0
        gdf_with_all_columns["dist_to_shop"] = 0
        gdf_with_all_columns["dist_to_school"] = 0
        gdf_with_all_columns["dist_to_industrial"] = 0
        gdf_with_all_columns["dist_to_recreation_ground"] = 0
        gdf_with_all_columns["dist_to_nature"] = 0
        gdf_with_all_columns["avg_neighbour_price"] = 0

        try:
            # Run the plot function
            assess.plot_all_parameters(gdf_with_all_columns)
        except Exception as e:
            # Fail the test if an exception is caught
            assert False, f"Function raised an exception: {e}"

    def test_pois_to_coords(self):
        pois = access.get_pois(53.4083112, -1.5238095, 0.1, {"amenity": True})
        res = assess.pois_to_coords(pois)
        
        assert str(res) == "POINT (-1.5231254 53.4080767)"

    def test_dist_to_poi(self):
        res = assess.dist_to_poi(Point(1.0, 1.0), {"latitude": 0.0, "longitude": 0.0})

        assert int(res) == 156

if __name__ == '__main__':
    unittest.main()