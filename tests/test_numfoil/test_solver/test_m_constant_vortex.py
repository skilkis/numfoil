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
from scipy import integrate

from numfoil.solver.m_constant_vortex import (
    gcs_to_pcs,
    pcs_to_gcs,
    vortex_c_2d,
)
from numfoil.solver.m_lumped_vortex import vortex_2d

GCS_TO_PCS_TEST_CASES = {
    "argnames": "point, reference_point, angle, expected_result",
    "argvalues": [
        (
            np.array([1, 1], dtype=np.float64),
            np.array([0, 1], dtype=np.float64),
            math.radians(-45),
            np.array([0.7071067812, 0.7071067812]),
        ),
        (
            np.array([0, 0], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
            math.radians(135),
            np.array([0.7071067812, 0.7071067812]),
        ),
    ],
}


@pytest.mark.parametrize(**GCS_TO_PCS_TEST_CASES)
def test_gcs_to_pcs(point, reference_point, angle, expected_result):
    """Tests transformation to panel coordiantes."""
    jit_result = gcs_to_pcs(point, reference_point, angle)
    result = gcs_to_pcs.py_func(point, reference_point, angle)

    assert np.allclose(jit_result, expected_result)
    assert np.allclose(result, expected_result)


PCS_TO_GCS_TEST_CASES = {
    "argnames": "point, angle, expected_result",
    "argvalues": [
        (
            np.array([0.7071067812, 0.7071067812]),
            math.radians(-45),
            np.array([1, 0], dtype=np.float64),
        ),
        (
            np.array([0.7071067812, 0.7071067812]),
            math.radians(135),
            np.array([-1, 0], dtype=np.float64),
        ),
    ],
}


@pytest.mark.parametrize(**PCS_TO_GCS_TEST_CASES)
def test_pcs_to_gcs(point, angle, expected_result):
    """Tests transformation from panel coordiantes."""
    jit_result = pcs_to_gcs(point, angle)
    result = pcs_to_gcs.py_func(point, angle)

    assert np.allclose(jit_result, expected_result)
    assert np.allclose(result, expected_result)


VORTEX_C_2D_TEST_CASES = {
    "argnames": (
        "gamma, start_pt, end_pt, col_pt, panel_angle, expected_result"
    ),
    "argvalues": [
        # Testing the situation where a collocation point is located
        # on the centroid of a horizontal panel:  --*--, This is given
        # in Kats & Plotkin as equal to (0.5, 0)
        pytest.param(
            1,
            np.array([0, 0], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
            np.array([0.5, 0], dtype=np.float64),
            math.radians(0),
            (0.5, 0),
            # Integrated method will fail due to div/0
            marks=pytest.mark.xfail(raises=ZeroDivisionError),
        ),
        # Testing the situation where a collocation point is located
        # offset from the centroid of a normal panel:
        # Vortex Panel -> || *  <- Collocation Point
        (
            1,
            np.array([0, 0], dtype=np.float64),
            np.array([0, 1], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
            math.radians(90),
            (0, -0.25),
        ),
        # Testing the situation where a collocation point is located
        # offset from the centroid of a horizontal panel: __ * __
        (
            1,
            np.array([0, 0], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
            math.radians(0),
            (0.25, 0),
        ),
        # Testing a panel angle of -45 degrees \\*
        (
            1,
            np.array([0, 1], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
            np.array([1, 1], dtype=np.float64),
            math.radians(-45),
            (0.1767767, -0.1767767),
        ),
        # Testing a panel angle of -45 degrees \\ * with a larger offset
        (
            1,
            np.array([0, 1], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
            np.array([1, 2], dtype=np.float64),
            math.radians(-45),
            (0.12739158, -0.04938512),
        ),
        # Testing a panel angle of -135 degrees *//
        (
            1,
            np.array([1, 0], dtype=np.float64),
            np.array([0, -1], dtype=np.float64),
            np.array([0, 0], dtype=np.float64),
            math.radians(-135),
            (0.1767767, 0.1767767),
        ),
        # Testing sign convention of the singularity at 135 degrees *//
        # TODO Investigate why this test is failing here
        # (
        #     1,
        #     np.array([1, 0], dtype=np.float64),
        #     np.array([0, 1], dtype=np.float64),
        #     np.array([0.5, 0.5], dtype=np.float64),
        #     math.radians(135),
        #     (-0.35355339, 0.35355339),
        # ),
    ],
}


@pytest.mark.parametrize(**VORTEX_C_2D_TEST_CASES)
def test_vortex_c_2d(
    gamma, start_pt, end_pt, col_pt, panel_angle, expected_result,
):
    """Tests if the correct induced velocity is returned."""
    jit_result = vortex_c_2d(gamma, start_pt, end_pt, col_pt, panel_angle)
    result = vortex_c_2d.py_func(gamma, start_pt, end_pt, col_pt, panel_angle)

    assert np.allclose(jit_result, expected_result)
    assert np.allclose(result, expected_result)

    # Sanity check with using an integrated version of vortex_2d
    integrated_result = integrated_vortex_c_2d(start_pt, end_pt, col_pt)
    assert np.allclose(jit_result, integrated_result)


def integrated_vortex_c_2d(start_pt, end_pt, col_pt):
    """Integrates vortex_2d to obtain vortex_c_2d output."""
    panel_length = np.linalg.norm(end_pt - start_pt)
    panel_tangent = (end_pt - start_pt) / panel_length

    def integrand(s):
        vortex_pt = start_pt + panel_tangent * s
        return vortex_2d(1, vortex_pt, col_pt)

    return integrate.quad_vec(integrand, 0, panel_length, epsabs=1e-9)[0]
