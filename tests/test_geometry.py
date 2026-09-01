import unittest

import numpy as np

from adapters.common.geometry import (
    fit_quadratic,
    point_errors,
    transform_points_homography,
    transform_points_quadratic,
)


class GeometryTests(unittest.TestCase):
    def test_identity_homography(self):
        points = np.array([[0.0, 0.0], [10.5, 4.25], [99.0, 120.0]])
        transformed = transform_points_homography(points, np.eye(3))
        np.testing.assert_allclose(transformed, points, atol=1e-10)
        np.testing.assert_allclose(point_errors(transformed, points), 0.0, atol=1e-10)

    def test_quadratic_round_trip_on_synthetic_correspondences(self):
        source = np.array([
            [0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0],
            [5.0, 2.0], [2.0, 5.0], [7.0, 8.0], [12.0, 4.0],
        ])
        coefficients = np.array([
            [3.0, -2.0], [1.1, 0.04], [-0.03, 0.9],
            [0.002, -0.001], [0.0005, 0.0008], [-0.0007, 0.0004],
        ])
        target = transform_points_quadratic(source, coefficients)
        estimated = fit_quadratic(source, target)
        np.testing.assert_allclose(estimated, coefficients, atol=1e-10)
        np.testing.assert_allclose(
            transform_points_quadratic(source, estimated), target, atol=1e-10
        )

    def test_quadratic_rejects_too_few_points(self):
        with self.assertRaises(ValueError):
            fit_quadratic(np.zeros((5, 2)), np.zeros((5, 2)))


if __name__ == "__main__":
    unittest.main()
