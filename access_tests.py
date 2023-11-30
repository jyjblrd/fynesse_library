#!/usr/bin/env python

import unittest
import fynesse.access as access

class AccessTests(unittest.TestCase):
    def test_get_pois(self):
        res = access.get_pois(53.4083112, -1.5238095, 5, {"amenity": True})
        assert res.size != 0

    def test_get_location_info(self):
        res = access.get_location_info(53.4083112, -1.5238095)
        assert res == ('Sheffield', 'United Kingdom', 'S6 4SE')


if __name__ == '__main__':
    unittest.main()