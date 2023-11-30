#!/usr/bin/env python

import unittest
import fynesse.helpers as helpers
import numpy as np

class AccessTests(unittest.TestCase):
    def test_km_to_lat_long(self):
        res = helpers.km_to_lat_long(100000, 10.0)
        correct = [899.3216059187305, 913.1950912937036]
        
        assert [round(x, 1) for x in res] == [round(x, 1) for x in correct]

    def test_gen_bounding_box(self):
        res = helpers.gen_bounding_box(10.0, 99.0, 20)
        correct = [10.089932160591873, 9.910067839408127, 99.09131950912938, 98.90868049087062]
        
        assert [round(x, 1) for x in res] == [round(x, 1) for x in correct]

    def test_normalize(self):
        res = helpers.normalize(np.array([1,2]))
        correct = np.array([0,1])

        assert all(res == correct)


if __name__ == '__main__':
    unittest.main()