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

from numfoil.solver.m_lumped_vortex import calc_lumped_vortex_im, vortex_2d

VORTEX_2D_TEST_CASES = {
    "argnames": "gamma, vortex_pt, col_pt, expected_result",
    "argvalues": [
        (
            2,
            np.array([[0, 0]], dtype=np.float64),
            np.array([[1, 0]], dtype=np.float64),
            np.array([[0, -1 / math.pi]]),
        )
    ],
}


@pytest.mark.parametrize(**VORTEX_2D_TEST_CASES)
def test_vortex_2d(gamma, vortex_pt, col_pt, expected_result):
    """Tests if the correct induced velocity is returned."""
    result = vortex_2d(gamma, vortex_pt, col_pt)
    jit_result = vortex_2d.py_func(gamma, vortex_pt, col_pt)

    assert np.allclose(result, expected_result)
    assert np.allclose(jit_result, expected_result)


offset_matrix = np.stack((np.arange(5) * 0.2, np.zeros(5)), axis=1)
CALC_LUMPED_VORTEX_IM_TEST_CASES = {
    "argnames": "vortex_pts, col_pts, panel_normals, expected_result",
    "argvalues": [
        (
            np.array([[0.2 * 0.25, 0]]) + offset_matrix,
            np.array([[0.2 * 0.75, 0]]) + offset_matrix,
            np.repeat(np.array([[0, 1]], dtype=np.float64), 5, axis=0),
            5
            / math.pi
            * np.array(
                [
                    [-1, 1, 1 / 3, 1 / 5, 1 / 7],
                    [-1 / 3, -1, 1, 1 / 3, 1 / 5],
                    [-1 / 5, -1 / 3, -1, 1, 1 / 3],
                    [-1 / 7, -1 / 5, -1 / 3, -1, 1],
                    [-1 / 9, -1 / 7, -1 / 5, -1 / 3, -1],
                ]
            ),
        )
    ],
}


@pytest.mark.parametrize(**CALC_LUMPED_VORTEX_IM_TEST_CASES)
def test_calc_lumpedvortex_im(
    vortex_pts, col_pts, panel_normals, expected_result
):
    """Tests if the lumped vortex influence matrix is correct."""
    jit_result = calc_lumped_vortex_im(vortex_pts, col_pts, panel_normals)
    result = calc_lumped_vortex_im.py_func(vortex_pts, col_pts, panel_normals)

    assert np.allclose(jit_result, expected_result)
    assert np.allclose(result, expected_result)
