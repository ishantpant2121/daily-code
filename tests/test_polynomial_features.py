"""Tests for polynomial feature expansion."""

import unittest
from preprocessing import PolynomialFeatures


class PolynomialFeaturesTests(unittest.TestCase):
    def test_generates_bias_powers_and_interactions_in_degree_order(self):
        self.assertEqual(
            PolynomialFeatures(2).fit_transform([[2, 3]]),
            [[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]],
        )

    def test_can_omit_bias_and_support_higher_degree(self):
        self.assertEqual(
            PolynomialFeatures(3, include_bias=False).fit_transform([[2]]),
            [[2.0, 4.0, 8.0]],
        )

    def test_checks_fitted_state_feature_width_and_options(self):
        with self.assertRaises(RuntimeError):
            PolynomialFeatures().transform([[1]])
        transformer = PolynomialFeatures().fit([[1, 2]])
        with self.assertRaises(ValueError):
            transformer.transform([[1]])
        for kwargs in ({"degree": 0}, {"degree": True}, {"include_bias": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PolynomialFeatures(**kwargs)


if __name__ == "__main__":
    unittest.main()
