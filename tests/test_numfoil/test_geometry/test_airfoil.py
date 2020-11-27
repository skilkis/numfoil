# Copyright 2020 Kilian Swannet, San Kilkis

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
from functools import cached_property
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from numfoil.geometry import Airfoil, NACA4Airfoil, UIUCAirfoil
from tests.test_numfoil.helpers import (
    ScenarioTestSuite,
    calc_curve_error,
    get_naca_airfoils,
)

AIRFOIL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestAirfoil:

    test_class = Airfoil

    def test_abc(self):
        """Tests if the ABC module works as expected."""

        # TODO update abstractmethods to actually reflect the Airfoil class
        # Getting abstract methods from current `test_class`
        abstractmethods = ["camber_at", "surfaces_at"]

        # Checking if TypeError is raised for all abstractmethods
        with pytest.raises(TypeError) as e:
            self.test_class()
            assert all(m in str(e.value) for m in abstractmethods)

    ENSURE_1D_VECTOR_TEST_CASES = {
        "argnames": "x, expected_result",
        "argvalues": [
            # Testing if float or int can be converted into an array
            (1, np.array([1])),
            # Checking that 1D numpy array passes
            (np.array([1, 2, 3]), np.array([1, 2, 3])),
            # Checking that 2D numpy array fails
            pytest.param(
                np.array([1, 2, 3]).reshape(1, 3),
                None,
                marks=pytest.mark.xfail(raises=ValueError),
            ),
        ],
    }

    @pytest.mark.parametrize(**ENSURE_1D_VECTOR_TEST_CASES)
    def test_ensure_1d_vector(self, x, expected_result):
        """Tests if inputs always yield 1D numpy arrays."""
        result = self.test_class.ensure_1d_vector(x)
        assert np.allclose(result, expected_result)


class TestNACA4Airfoil:

    test_class = NACA4Airfoil
    atol = 1e-4  # Absolute tolerance used for numerical error based tests

    VALUE_ATTRIBUTES_TEST_CASES = {
        "argnames": "name, expected_attributes",
        "argvalues": [
            (
                "naca0012",
                {"max_camber": 0, "camber_location": 0, "max_thickness": 0.12},
            ),
            (
                "naca0102",
                {"max_camber": 0, "camber_location": 0, "max_thickness": 0.02},
            ),
            (
                "naca1012",
                {
                    "max_camber": 0.01,
                    "camber_location": 0.1,
                    "max_thickness": 0.12,
                },
            ),
            (
                "naca2312",
                {
                    "max_camber": 0.02,
                    "camber_location": 0.3,
                    "max_thickness": 0.12,
                },
            ),
        ],
    }

    @pytest.mark.parametrize(**VALUE_ATTRIBUTES_TEST_CASES)
    def test_value_attributes(self, name, expected_attributes):
        """Tests if the ``name`` input is stored correctly in attrs."""
        airfoil = self.test_class(name)
        for attr_name, expected_result in expected_attributes.items():
            result = getattr(airfoil, attr_name)
            assert result == expected_result

    CAMBERED_TEST_CASES = {
        "argnames": "name, expected_result",
        "argvalues": [("0012", False), ("2312", True)],
    }

    @pytest.mark.parametrize(**CAMBERED_TEST_CASES)
    def test_cambered(self, name, expected_result):
        """Tests if airfoil cambere is correctly detected."""
        airfoil = self.test_class(name)
        assert airfoil.cambered == expected_result

    CAMBERLINE_AT_TEST_CASES = {
        "argnames": "name, x, expected_result",
        "argvalues": [
            (
                "naca0012",
                np.array([0, 0.5, 1.0]),
                np.array([[0, 0], [0.5, 0], [1.0, 0]]),
            ),
            (
                "naca2512",
                np.array([0, 0.25, 0.5, 0.75, 1.0]),
                np.array(
                    [
                        [0, 0],
                        [0.25, 0.015],
                        [0.5, 0.02],
                        [0.75, 0.015],
                        [1.0, 0],
                    ]
                ),
            ),
        ],
    }

    @pytest.mark.parametrize(**CAMBERLINE_AT_TEST_CASES)
    def test_camberline_at(self, name, x, expected_result):
        """Tests if the camber-line returns correct values."""
        airfoil = self.test_class(name)
        result = airfoil.camberline_at(x)
        assert np.allclose(result, expected_result)

    @pytest.mark.parametrize("name", ["naca0012", "naca2312"])
    def test_camber_tangent_at(self, name):
        """Tests camber tangent vectors for correct orientation."""
        airfoil = self.test_class(name)

        tan_max_camber = airfoil.camber_tangent_at(airfoil.camber_location)
        assert np.allclose(tan_max_camber, np.array([1, 0]))

        tan_le = airfoil.camber_tangent_at(0)
        tan_te = airfoil.camber_tangent_at(1)
        expected_tan_le = (
            np.array([1, 1]) if airfoil.cambered else np.array([1, 0])
        )
        expected_tan_te = (
            np.array([1, -1]) if airfoil.cambered else np.array([1, 0])
        )
        assert np.allclose(np.sign(tan_le), expected_tan_le)
        assert np.allclose(np.sign(tan_te), expected_tan_te)

    @pytest.mark.parametrize("name", ["naca0012", "naca2312"])
    def test_camber_normal_at(self, name):
        """Tests camber normal vectors for correct orientation."""
        airfoil = self.test_class(name)

        tan_max_camber = airfoil.camber_normal_at(airfoil.camber_location)
        assert np.allclose(tan_max_camber, np.array([0, 1]))

        tan_le = airfoil.camber_normal_at(0)
        tan_te = airfoil.camber_normal_at(1)
        expected_tan_le = (
            np.array([-1, 1]) if airfoil.cambered else np.array([0, 1])
        )
        expected_tan_te = (
            np.array([1, 1]) if airfoil.cambered else np.array([0, 1])
        )
        assert np.allclose(np.sign(tan_le), expected_tan_le)
        assert np.allclose(np.sign(tan_te), expected_tan_te)

    SURFACE_AT_TEST_CASES = {
        "argnames": "name, te_closed, x, expected_result",
        "argvalues": [
            (
                "naca0012",
                True,
                np.array([0, 0.3, 1]),
                np.array([[0, 0], [0.3, 0.06], [1, 0]]),
            ),
            (
                "naca0014",
                True,
                np.array([0, 0.3, 1]),
                np.array([[0, 0], [0.3, 0.07], [1, 0]]),
            ),
        ],
    }

    @pytest.mark.parametrize(**SURFACE_AT_TEST_CASES)
    def test_surfaces_at(self, name, te_closed, x, expected_result):
        """Tests upper and lower surfaces at max ordinate and le/te."""
        airfoil = self.test_class(name, te_closed=te_closed)
        pts_upper = airfoil.upper_surface_at(x)
        pts_lower = airfoil.lower_surface_at(x)

        if isinstance(expected_result, (list, tuple)):
            expected_upper, expected_lower = expected_result
        else:
            expected_upper = expected_result
            expected_lower = expected_upper * np.array([[1, -1]])

        assert np.allclose(pts_upper, expected_upper, atol=self.atol)
        assert np.allclose(pts_lower, expected_lower, atol=self.atol)

    OFFSET_VECTORS_TEST_CASES = {
        "argnames": "name, te_closed, x, expected_result",
        "argvalues": [
            (
                "naca0012",
                True,
                np.array([0, 0.3, 1]),
                np.array([[0, 0], [0, 0.06], [0, 0]]),
            ),
            (
                "naca0014",
                True,
                np.array([0, 0.3, 1]),
                np.array([[0, 0], [0, 0.07], [0, 0]]),
            ),
        ],
    }

    @pytest.mark.parametrize(**OFFSET_VECTORS_TEST_CASES)
    def test_offset_vectors_at(self, name, te_closed, x, expected_result):
        """Tests that offset vector is half thickness at x=0.3."""
        airfoil = self.test_class(name, te_closed=te_closed)
        result = airfoil.offset_vectors_at(x)
        assert np.allclose(result, expected_result, atol=self.atol)

    @pytest.mark.parametrize(
        "name, te_closed", [("0012", True), ("0012", False)]
    )
    def test_half_thickness_at(self, name, te_closed):
        """Testing half thickness at max ordinate and le/te edges."""
        airfoil = self.test_class(name, te_closed=te_closed)

        result = airfoil.half_thickness_at(0.3)
        expected = airfoil.max_thickness / 2
        assert np.allclose(result, expected, atol=self.atol)
        if te_closed:
            assert np.allclose(airfoil.half_thickness_at(1), 0)
        else:
            assert airfoil.half_thickness_at(1) != 0

    PARSE_NACA_CODE_TEST_CASES = {
        "argnames": "code, expected_result",
        "argvalues": [
            ("naca0012", (0, 0, 1, 2)),
            ("NACA2312", (2, 3, 1, 2)),
            ("0015", (0, 0, 1, 5)),
            pytest.param(
                "naca", None, marks=pytest.mark.xfail(raises=ValueError)
            ),
            pytest.param(
                "naca2412xxxx",
                None,
                marks=pytest.mark.xfail(raises=ValueError),
            ),
        ],
    }

    @pytest.mark.parametrize(**PARSE_NACA_CODE_TEST_CASES)
    def test_parse_naca_code(self, code, expected_result):
        """Tests if a naca code can be converted to a map of ints."""
        result = self.test_class.parse_naca_code(code)
        assert isinstance(result, map)
        assert tuple(result) == expected_result

    PLOT_TEST_CASES = {
        "argnames": "name, n_points, show",
        "argvalues": [("naca0012", 100, False), ("naca2312", 150, False)],
    }

    @pytest.mark.parametrize(**PLOT_TEST_CASES)
    def test_plot(self, name, n_points, show):
        """Tests if inputs work and all plot elements are rendered."""
        airfoil = self.test_class(name)
        fig, ax = airfoil.plot(n_points=n_points, show=show)

        # Testing if plot contains correct number of points and lines
        assert len(ax.lines) >= 3 if airfoil.cambered else 2
        for line in ax.lines:
            # Testing _x is sufficient since _y must be equal length
            assert len(line._x) == n_points

        # Testing Legend
        legend_texts = [t._text for t in ax.get_legend().texts]
        if airfoil.cambered:
            required_texts = ("Upper Surface", "Lower Surface", "Camber Line")
        else:
            required_texts = ("Upper Surface", "Lower Surface")
        assert all(t in legend_texts for t in required_texts)

    @pytest.mark.parametrize("filename", get_naca_airfoils(AIRFOIL_DATA_DIR))
    def test_airfoil_accuracy(self, filename):
        """Tests accuracy of surfaces with UIUC NACA 4 series files."""
        fname = os.path.join(AIRFOIL_DATA_DIR, filename)
        naca_code = os.path.splitext(filename)[0]
        expected_pts = np.loadtxt(fname, comments="NACA")
        le_idx = np.argwhere(expected_pts == np.array([0, 0]))[0, 0]

        # Fetching airfoil points at same x-value as data file
        a = self.test_class(naca_code, te_closed=expected_pts[0, 1] == 0)
        result_pts = np.zeros(expected_pts.shape)
        result_pts[:le_idx] = a.upper_surface_at(expected_pts[:le_idx, 0])
        result_pts[le_idx:] = a.lower_surface_at(expected_pts[le_idx:, 0])

        if a.cambered:
            assert calc_curve_error(result_pts, expected_pts) <= self.atol
        else:
            # If the airfoil is symmetric points are compared directly
            np.allclose(result_pts, expected_pts)


class FileAirfoilTester(ScenarioTestSuite):

    test_class = None
    atol = 1e-4  # Absolute tolerance used for numerical error based tests

    def test_points(self, scenario):
        """Tests if airfoil points can be correctly parsed."""
        obj, _ = scenario
        points = obj.points
        _, n_dims = points.shape

        # Checking that row-vectors were parsed in
        assert n_dims == 2

        # Checking that start and end coordinates are at x=1
        assert np.allclose(points[(0, -1), 0], 1, atol=self.atol)

        # Checking that the leading edge is located at x=0
        assert np.allclose(np.min(points, axis=0)[0], 0, atol=self.atol)

    def test_le_idx(self, scenario):
        """Ensures that the leading-edge index corresponds to x=0."""
        obj, _ = scenario
        assert obj.points[obj.le_idx][0] == 0


class TestUIUCAirfoil(FileAirfoilTester):

    test_class = UIUCAirfoil
    airfoil_data_dir = AIRFOIL_DATA_DIR

    @cached_property
    def SCENARIOS(self) -> Dict[str, Airfoil]:
        """Creates scenarioes using NACA 4 series files."""
        return {
            a: self.test_class(Path(self.airfoil_data_dir) / a)
            for a in get_naca_airfoils(self.airfoil_data_dir)
        }

    EXPECTED_CAMBERED = {
        "naca0008": False,
        "naca0018": False,
        "naca1410": True,
        "naca2412": True,
        "naca4412": True,
    }

    def test_cambered(self, scenario):
        """Tests if camber is correctly identified."""
        obj, label = scenario
        airfoil_name = Path(label).stem
        assert obj.cambered == NACA4Airfoil(airfoil_name).cambered
