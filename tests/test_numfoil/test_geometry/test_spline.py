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

from numfoil.geometry.spline import BSpline2D
from tests.test_numfoil.helpers import ScenarioTestSuite


class TestBSpline2D(ScenarioTestSuite):

    SCENARIOS = {
        "line": BSpline2D(
            np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
        ),
        "reversed_line": BSpline2D(
            np.array([[3, 0], [2, 0], [1, 0], [0, 0]], dtype=np.float64)
        ),
    }

    EXPECTED_EVALUATE_AT = {
        "line": np.array([[0, 0], [1.5, 0], [3.0, 0]]),
        "reversed_line": np.array(([3.0, 0], [1.5, 0], [0, 0])),
    }

    def test_evaluate_at(self, scenario):
        """Tests if points on the spline are evaluated correctly."""
        obj, label = scenario
        result = obj.evaluate_at(np.array([0, 0.5, 1], dtype=np.float64))
        assert np.allclose(result, self.EXPECTED_EVALUATE_AT[label])

    EXPECTED_TANGENT_AT = {
        "line": np.array([[1, 0]], dtype=np.float64),
        "reversed_line": np.array([[-1, 0]], dtype=np.float64),
    }

    def test_tangent_at(self, scenario):
        """Tests if the correct tangent vector is returned."""
        obj, label = scenario
        result = obj.tangent_at(0.5)
        assert np.allclose(result, self.EXPECTED_TANGENT_AT[label])

    EXPECTED_NORMAL_AT = {
        "line": np.array([[0, 1]], dtype=np.float64),
        "reversed_line": np.array([[0, -1]], dtype=np.float64),
    }

    def test_normal_at(self, scenario):
        """Tests if the correct normal vector is returned."""
        obj, label = scenario
        result = obj.normal_at(0.5)
        assert np.allclose(result, self.EXPECTED_NORMAL_AT[label])
