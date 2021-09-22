import unittest
import numpy as np


def assert_arrays_almost_equal(test: unittest.TestCase, actuals: np.ndarray, desires: np.ndarray, dif_ratio=1e-3, limit=1e-8):
    assert(isinstance(actuals, np.ndarray))
    assert(isinstance(desires, np.ndarray))

    test.assertEqual(len(actuals.shape), len(desires.shape))

    for _, (a, b) in enumerate(zip(actuals.shape, desires.shape)):
        test.assertEqual(a, b)

    for _, (input, ref) in enumerate(zip(actuals.ravel(), desires.ravel())):
        delta = max(abs(ref * dif_ratio), limit)
        test.assertAlmostEqual(input, ref, delta=delta)
