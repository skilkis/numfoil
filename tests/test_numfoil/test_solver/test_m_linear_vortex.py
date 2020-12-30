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

from pathlib import Path

import numpy as np
import pytest

from numfoil.geometry import NACA4Airfoil, Panel2D
from numfoil.solver.base import PanelMethod
from numfoil.solver.m_linear_vortex import (
    calc_integration_constants,
    calc_linear_vortex_im,
    calc_vortex_coefficients,
)

REL_TOL = 5e-5
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def naca2412():
    """Returns a NACA2412 :py:class:`NACA4Airfoil` object."""
    return NACA4Airfoil(naca_code="naca2412", te_closed=True)


@pytest.fixture(scope="module")
def panels(naca2412):
    """Returns a paneled version of a NACA2412 airfoil."""
    sample_u = PanelMethod.get_sample_parameters(num=6, spacing="cosine")
    bot_pts = naca2412.lower_surface_at(sample_u[::-1])
    top_pts = naca2412.upper_surface_at(sample_u[1:])
    return Panel2D(np.vstack((bot_pts, top_pts)))


def test_calc_integration_constants(panels):
    """Tests that integration constants are calculated correctly.

    The test case calculates the coefficients due to the influence of
    vortex (n - 1) on panel (n)
    """
    expected_result = dict(
        a=-0.30074680202253584,
        b=0.0904534136839456,
        c=-0.045227355204173834,
        d=0.9989767196192489,
        e=0.0021851217730942327,
        f=-3.6571407713042885,
        g=0.03797726355934574,
        p=-0.015784868222446846,
        q=0.30034022644185643,
    )
    for i, panel in enumerate(panels):
        panel
    second_to_last_panel = panels[-2]
    last_panel = panels[-1]
    start_pt, _ = second_to_last_panel.nodes
    kwargs = dict(
        col_pt=last_panel.points_at(0.5)[0, :],
        vort_pt=start_pt[0, :],
        col_panel_angle=last_panel.angles[0, 0],
        vort_panel_angle=second_to_last_panel.angles[0, 0],
        vort_panel_length=second_to_last_panel.lengths[0, 0],
    )

    jit_result = calc_integration_constants(**kwargs)
    result = calc_integration_constants.py_func(**kwargs)

    for i, coef in enumerate(expected_result.keys()):
        assert jit_result[i] == pytest.approx(
            expected_result[coef], rel=REL_TOL
        )
        assert result[i] == pytest.approx(expected_result[coef], rel=REL_TOL)


CALC_VORTEX_COEFFICIENTS_TEST_CASES = {
    # Expected result is of the form (cn_1, cn_2, ct_1, ct_2)
    "argnames": "col_idx, vort_idx, expected_result",
    "argvalues": [
        # Testing influence of vortex gamma_2 on collocation point 1
        (0, 1, (1.17573979, 0.64921316, 3.42384684e-03, 2.98904500e-03)),
        # Testing influence of vortex gamma_10 on collocation point 1
        (0, 9, (0.31826418, -0.95201426, 1.40537373e00, 1.13522801e00)),
        # Testing influence of vortex gamma_3 on collocation point 8
        (7, 2, (0.44829642, -0.25604628, 9.64947195e-01, 1.01300764e00)),
    ],
}


@pytest.mark.parametrize(**CALC_VORTEX_COEFFICIENTS_TEST_CASES)
def test_calc_vortex_coefficients(panels, col_idx, vort_idx, expected_result):
    """Tests influence coefficient results for various panels."""

    col_panel = panels[col_idx]
    vort_panel = panels[vort_idx]
    kwargs = dict(
        col_pt=col_panel.points_at(0.5)[0, :],
        vort_pt=vort_panel.nodes[0][0, :],
        col_panel_angle=col_panel.angles[0, 0],
        vort_panel_angle=vort_panel.angles[0, 0],
        vort_panel_length=vort_panel.lengths[0, 0],
    )

    jit_result = calc_vortex_coefficients(**kwargs)
    result = calc_vortex_coefficients.py_func(**kwargs)

    assert jit_result == pytest.approx(expected_result, rel=REL_TOL)
    assert result == pytest.approx(expected_result, rel=REL_TOL)


def test_calc_linear_vortex_im(panels):
    """Tests influence matrix values against verification data."""
    kwargs = dict(
        col_pts=panels.points_at(0.5),
        vort_pts=panels.nodes[0],  # Start nodes
        panel_angles=panels.angles,
        panel_lengths=panels.lengths,
    )

    jit_result = calc_linear_vortex_im(**kwargs)
    result = calc_linear_vortex_im.py_func(**kwargs)

    an = np.load(DATA_DIR / "an_matrix_naca2412.npy")
    at = np.load(DATA_DIR / "at_matrix_naca2412.npy")

    assert jit_result[0, :-1, :-1] == pytest.approx(an[:-1, :-1], rel=REL_TOL)
    assert jit_result[1, :-1, :-1] == pytest.approx(at[:-1, :-1], rel=REL_TOL)

    assert result[0, :-1, :-1] == pytest.approx(an[:-1, :-1], rel=REL_TOL)
    assert result[1, :-1, :-1] == pytest.approx(at[:-1, :-1], rel=REL_TOL)
