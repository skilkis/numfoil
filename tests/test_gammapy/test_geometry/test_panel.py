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

import numpy as np
import pytest
from matplotlib.quiver import Quiver

from gammapy.geometry.panel import Panel2D
from tests.test_gammapy.helpers import ScenarioTestSuite


class TestPanel2D(ScenarioTestSuite):

    SCENARIOS = {
        "plate": Panel2D([(0, 0), (1, 0)]),
        "trapezoid": Panel2D([(0, 0), (1, 1), (2, 1), (3, 0)]),
        "bucket": Panel2D([(3, 0), (2, 0), (2, -1), (1, -1), (1, 0), (-1, 0)]),
    }

    INITIALIZATION_TEST_CASES = {
        "argnames": "array, expected_result",
        "argvalues": [
            # Testing if a list works properly as an input
            ([[0, 0], [1, 1]], np.array([[0, 0], [1, 1]])),
            # Testing that 1 element array raises an error
            pytest.param(
                np.array([[1, 1]]),
                None,
                marks=pytest.mark.xfail(raises=ValueError),
            ),
            # Checking that a column vector raises an error
            pytest.param(
                np.array([[0, 0], [1, 1], [2, 3]]).T,
                None,
                marks=pytest.mark.xfail(raises=ValueError),
            ),
        ],
    }

    @pytest.mark.parametrize(**INITIALIZATION_TEST_CASES)
    def test_initialization(self, array, expected_result):
        """Tests if :py:class:`Panel2D` is correctly instantiated."""
        result = Panel2D(array)
        assert np.allclose(result, expected_result)

        # Checking that the instantiated object is our derived class
        assert isinstance(result, Panel2D)

        # Ensuring that the properties/methods are available in result
        for attr in ("nodes", "tangents", "normals", "lengths", "points_at"):
            assert hasattr(result, attr)

    EXPECTED_NODES = {
        "plate": (np.array([[0, 0]]), np.array([[1, 0]])),
        "trapezoid": (
            np.array([[0, 0], [1, 1], [2, 1]]),
            np.array([[1, 1], [2, 1], [3, 0]]),
        ),
        "bucket": (
            np.array([[3, 0], [2, 0], [2, -1], [1, -1], [1, 0]]),
            np.array([[2, 0], [2, -1], [1, -1], [1, 0], [-1, 0]]),
        ),
    }

    def test_nodes(self, scenario):
        """Tests if the start and end nodes are correct."""
        result = scenario.obj.nodes
        expected = self.EXPECTED_NODES[scenario.label]
        assert all(np.allclose(r, e) for r, e in zip(result, expected))

    SQ_22 = np.sqrt(2) / 2  # x and y vector components of a 45 deg panel
    EXPECTED_TANGENTS = {
        "plate": np.array([[1, 0]]),
        "trapezoid": np.array([[SQ_22, SQ_22], [1, 0], [SQ_22, -SQ_22]]),
        "bucket": np.array([[-1, 0], [0, -1], [-1, 0], [0, 1], [-1, 0]]),
    }

    def test_tangents(self, scenario):
        """Tests the tangent vectors panel."""
        result = scenario.obj.tangents
        expected = self.EXPECTED_TANGENTS[scenario.label]
        assert np.allclose(result, expected)

    EXPECTED_NORMALS = {
        "plate": np.array([[0, 1]]),
        "trapezoid": np.array([[-SQ_22, SQ_22], [0, 1], [SQ_22, SQ_22]]),
        "bucket": np.array([[0, -1], [1, 0], [0, -1], [-1, 0], [0, -1]]),
    }

    def test_normals(self, scenario):
        """Tests the normal vectors of each panel."""
        result = scenario.obj.normals
        expected = self.EXPECTED_NORMALS[scenario.label]
        assert np.allclose(result, expected)

    EXPECTED_LENGTHS = {
        "plate": np.array([[1]]),
        "trapezoid": np.array([SQ_22 * 2, 1, SQ_22 * 2]).reshape(3, 1),
        "bucket": np.array([1, 1, 1, 1, 2]).reshape(5, 1),
    }

    def test_lengths(self, scenario):
        """Tests the length of each panel."""
        result = scenario.obj.lengths
        expected = self.EXPECTED_LENGTHS[scenario.label]
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("u", (0, 0.25, 0.5, 0.75, 1.0))
    def test_points_at(self, scenario, u):
        """Tests if points are on the panel at the right location."""
        pts = scenario.obj.points_at(u)
        starts, ends = scenario.obj.nodes

        # Testing if point is on panel and if u is correct using
        # The length of line segments method: A-C------B:
        # https://stackoverflow.com/a/17693146/11989587
        for pt, start, end, in zip(pts, starts, ends):
            length_AC = np.linalg.norm(pt - start)
            length_BC = np.linalg.norm(end - pt)
            length_AB = np.linalg.norm(end - start)

            # Testing if pt is on the panel
            assert length_AC + length_BC == pytest.approx(length_AB)

            # Testing if the pt corresponds to the correct u parameter
            assert length_AC / length_AB == pytest.approx(u)

    def test_plot(self, scenario):
        """Tests if inputs work and all plot elements are rendered."""
        fig, ax = scenario.obj.plot(show=False)

        # Ensure that both normal and tangent are drawn
        quivers = [c for c in ax.collections if isinstance(c, Quiver)]
        assert len(quivers) == 2

        # Testing if nodes and edges exist in legend
        legend_texts = [t._text for t in ax.get_legend().texts]
        required_texts = ("Nodes", "Edges")
        assert all(t in legend_texts for t in required_texts)

        # Making sure that the axis is equal which is important
        # for the direction of the Quiver arrows to rendered correctly
        assert ax._aspect == "equal"
