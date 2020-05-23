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

import math

import numpy as np
import pytest

from gammapy.solver.method import PanelMethod


class TestPanelMethod:

    SAMPLE_PARAMETERS_CASES = {
        "argnames": "func_args, expected_result",
        "argvalues": [
            # Checking if n_points = 0 works for both "cosine" case
            # Also checks if spacing is indeed an optional input
            ((0,), np.array([])),
            # Checking if n_points = 0 for "linear" case
            ((0, "linear"), np.array([])),
            # Checking cosine spacing with n_points = 5
            # Also checks if the input "cosine" functions as expected
            ((5, "cosine"), np.array([0, 0.14644, 0.5, 0.85355, 1])),
            # Checking if linear spacing works with n_points = 5
            ((5, "linear"), np.array([0, 0.25, 0.5, 0.75, 1])),
            # Checking if ValueError is raised with invalid spacing
            pytest.param(
                (0, None), None, marks=pytest.mark.xfail(raises=ValueError)
            ),
        ],
    }

    @pytest.mark.parametrize(**SAMPLE_PARAMETERS_CASES)
    def test_get_sample_parameters(self, func_args, expected_result):
        """Tests if parameter sampling functions correctly."""
        result = PanelMethod.get_sample_parameters(*func_args)
        assert np.allclose(result, expected_result, atol=1e-5)

    @pytest.mark.parametrize("attribute", ["airfoil", "n_panels", "spacing"])
    def test_settable(self, attribute, monkeypatch):
        """Tests that users can't set reserved attributes."""
        # Temporarily disable ABC module with monkeypatch
        monkeypatch.delattr(PanelMethod, "__abstractmethods__")
        mock_airfoil = object()
        obj = PanelMethod(airfoil=mock_airfoil, n_panels=200, spacing="linear")
        with pytest.raises(AttributeError):
            setattr(obj, attribute, None)

    GET_VELOCITY_VECTORS_TEST_CASES = {
        "argnames": "velocity, alpha, expected_result",
        "argvalues": [
            (20, 0, np.array([[20, 0]])),
            (20, 45, np.array([[math.sqrt(200), math.sqrt(200)]])),
            (20, -90, np.array([[0, -20]])),
            # Testing if alpha can be a sequence of floats
            (
                20,
                [0, -45, 135],
                np.array(
                    [
                        [20, 0],
                        [math.sqrt(200), -math.sqrt(200)],
                        [-math.sqrt(200), math.sqrt(200)],
                    ]
                ),
            ),
        ],
    }

    @pytest.mark.parametrize(**GET_VELOCITY_VECTORS_TEST_CASES)
    def test_get_velocity_vectors(self, velocity, alpha, expected_result):
        """Tests conversion of angles to velocity vectors."""
        result = PanelMethod.get_velocity_vector(velocity, alpha)
        assert np.allclose(result, expected_result)
