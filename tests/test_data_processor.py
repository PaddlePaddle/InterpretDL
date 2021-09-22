import unittest
import numpy as np
from numpy.core.fromnumeric import size

import interpretdl as it
from interpretdl.data_processor.readers import *
from interpretdl.data_processor.visualizer import *
from interpretdl.data_processor.visualizer import _grayscale, _heatmap
from tests.utils import assert_arrays_almost_equal


class TestBasicMethods(unittest.TestCase):
    def test_random_seed(self):
        np.random.seed(42)

        desired = np.array([51, 92, 14, 71, 60, 20, 82, 86, 74, 74])
        results = np.random.randint(0, 100, size=10)
        assert_arrays_almost_equal(self, results, desired)


class TestImageProcessingMethods(unittest.TestCase):

    def test_resize_image(self):
        img = np.array(range(6*4*3), dtype=np.float32).reshape((6, 4, 3))
        result = resize_image(img, 2)
        desired = np.array([
            [[ 7.5,  8.5,  9.5], [13.5, 14.5, 15.5]],
            [[31.5, 32.5, 33.5], [37.5, 38.5, 39.5]],
            [[55.5, 56.5, 57.5], [61.5, 62.5, 63.5]]
        ])

        assert_arrays_almost_equal(self, result, desired)

    def test_crop_image(self):
        img = np.array(range(6*4*3), dtype=np.float32).reshape((6, 4, 3))
        result = crop_image(img, 2)
        desired = np.array([
            [[27., 28., 29.],  [30., 31., 32.]], 
            [[39., 40., 41.], [42., 43., 44.]]
        ])

        assert_arrays_almost_equal(self, result, desired)

    def test_preprocess_image(self):
        img = np.array(range(1*6*4*3), dtype=np.float32).reshape((1, 6, 4, 3))
        result = preprocess_image(img)
        desired = np.array([[[[-2.117904,   -2.0665295,  -2.0151553,  -1.9637811 ],
            [-1.9124068,  -1.8610326,  -1.8096583,  -1.7582841 ],
            [-1.7069098,  -1.6555356,  -1.6041613,  -1.5527871 ],
            [-1.5014127,  -1.4500386,  -1.3986642,  -1.34729   ],
            [-1.2959157,  -1.2445415,  -1.1931672,  -1.1417929 ],
            [-1.0904187,  -1.0390444,  -0.9876701,  -0.93629587]],

            [[-2.0182073,  -1.9656863,  -1.9131653,  -1.8606442 ],
            [-1.8081232,  -1.7556022,  -1.7030813,  -1.6505603 ],
            [-1.5980393,  -1.5455183,  -1.4929973,  -1.4404761 ],
            [-1.3879551,  -1.3354341,  -1.2829131,  -1.2303921 ],
            [-1.1778711,  -1.1253501,  -1.0728291,  -1.0203081 ],
            [-0.9677871,  -0.91526604, -0.86274505, -0.81022406]],

            [[-1.769586,   -1.7172984,  -1.6650108,  -1.6127232 ],
            [-1.5604357,  -1.5081481,  -1.4558605,  -1.4035729 ],
            [-1.3512853,  -1.2989978,  -1.2467102,  -1.1944226 ],
            [-1.142135,   -1.0898474,  -1.0375599,  -0.98527235],
            [-0.93298477, -0.8806972,  -0.8284096,  -0.77612203],
            [-0.72383446, -0.6715468,  -0.61925924, -0.56697166]]]
        ])

        assert_arrays_almost_equal(self, result, desired, 1e-6)


    def test_restore_image(self):
        f_img = np.array([[[[-2.117904,   -2.0665295,  -2.0151553,  -1.9637811 ],
            [-1.9124068,  -1.8610326,  -1.8096583,  -1.7582841 ],
            [-1.7069098,  -1.6555356,  -1.6041613,  -1.5527871 ],
            [-1.5014127,  -1.4500386,  -1.3986642,  -1.34729   ],
            [-1.2959157,  -1.2445415,  -1.1931672,  -1.1417929 ],
            [-1.0904187,  -1.0390444,  -0.9876701,  -0.93629587]],

            [[-2.0182073,  -1.9656863,  -1.9131653,  -1.8606442 ],
            [-1.8081232,  -1.7556022,  -1.7030813,  -1.6505603 ],
            [-1.5980393,  -1.5455183,  -1.4929973,  -1.4404761 ],
            [-1.3879551,  -1.3354341,  -1.2829131,  -1.2303921 ],
            [-1.1778711,  -1.1253501,  -1.0728291,  -1.0203081 ],
            [-0.9677871,  -0.91526604, -0.86274505, -0.81022406]],

            [[-1.769586,   -1.7172984,  -1.6650108,  -1.6127232 ],
            [-1.5604357,  -1.5081481,  -1.4558605,  -1.4035729 ],
            [-1.3512853,  -1.2989978,  -1.2467102,  -1.1944226 ],
            [-1.142135,   -1.0898474,  -1.0375599,  -0.98527235],
            [-0.93298477, -0.8806972,  -0.8284096,  -0.77612203],
            [-0.72383446, -0.6715468,  -0.61925924, -0.56697166]]]
        ])
        result = restore_image(f_img)
        desired = np.array(range(1*6*4*3), dtype=np.uint8).reshape((1, 6, 4, 3))

        assert_arrays_almost_equal(self, result, desired)


class TestVisualizeMethods(unittest.TestCase):

    def test_grayscale(self):
        img = np.array(range(5*5), dtype=np.float32).reshape((5, 5)) / 25.0
        result = _grayscale(img)
        desired = np.array([
            [  0,  10,  21,  32,  42],
            [ 53,  64,  75,  85,  96],
            [107, 118, 128, 139, 150],
            [160, 171, 182, 193, 203],
            [214, 225, 236, 246, 255]
        ])

        assert_arrays_almost_equal(self, result, desired)

    def test_overlay_grayscale(self):
        img = np.array(range(3*3*3), dtype=np.uint8).reshape((3, 3, 3))
        exp = np.array(range(3*3), dtype=np.float32).reshape((3, 3)) / 9.0
        result = overlay_grayscale(img, exp)
        desired = np.array([
            [[  0,   0,   1],
            [  1,  15,   3],
            [  3,  29,   4]],
            [[  5,  44,   6],
            [  7,  59,   8],
            [  9,  73,  10]],
            [[ 10,  88,  12],
            [ 12, 103,  13],
            [ 14, 117,  15]]
        ])

        assert_arrays_almost_equal(self, result, desired)

    def test_heatmap(self):
        exp = np.array(range(3*3), dtype=np.float32).reshape((3, 3)) / 9.0
        result = _heatmap(exp, resize_shape=(4, 4))
        desired = np.array([[[  0,   0, 128],
        [  0,   0, 204],
        [  0,  44, 255],
        [  0, 124, 255]],

       [[  0, 108, 255],
        [  0, 188, 255],
        [ 30, 255, 226],
        [110, 255, 146]],

       [[142, 255, 114],
        [222, 255,  34],
        [255, 192,   0],
        [255, 112,   0]],

       [[255, 128,   0],
        [255,  48,   0],
        [208,   0,   0],
        [128,   0,   0]]])

        assert_arrays_almost_equal(self, result, desired)


    def test_overlay_heatmap(self):
        img = np.array(range(3*3*3), dtype=np.uint8).reshape((3, 3, 3))
        exp = np.array(range(3*3), dtype=np.float32).reshape((3, 3)) / 9.0
        result = overlay_heatmap(img, exp)
        desired = np.array([[[  0,   0,  52],
        [  1,   2, 103],
        [  3,  53, 106]],

       [[  5, 106, 108],
        [ 57, 109,  60],
        [110, 111,  10]],

       [[112,  62,  12],
        [114,  13,  13],
        [ 65,  15,  15]]])

        assert_arrays_almost_equal(self, result, desired)

    def test_overlay_threshold(self):
        img = np.array(range(3*3*3), dtype=np.uint8).reshape((3, 3, 3))
        exp = np.array(range(3*3), dtype=np.float32).reshape((3, 3)) / 9.0
        result = overlay_threshold(img, exp)
        desired = np.array([[[  0,   0,   0],
        [  1,  18,   2],
        [  2,  36,   3]],

       [[  3,  55,   4],
        [  4,  73,   5],
        [  6,  91,   6]],

       [[  7, 109,   8],
        [  8, 127,   9],
        [  9, 145,  10]]])

        assert_arrays_almost_equal(self, result, desired)


if __name__ == '__main__':
    unittest.main()