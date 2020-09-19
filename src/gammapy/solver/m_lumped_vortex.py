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

"""Implements a First Order Lumped Vortex Panel Method."""

import math
from functools import cached_property
from typing import Type

import numba
import numpy as np

from gammapy.geometry.panel import Panel2D
from gammapy.solver.base import (
    BASE_NUMBA_CONFIG,
    PanelMethod,
    ThinFlowSolution,
)

AFFINE_90_CW = np.array([[0, -1], [1, 0]], dtype=np.float64)


class LumpedVortex(PanelMethod):
    @cached_property
    def panels(self) -> Panel2D:
        """Returns panels on the camber-line of the airfoil.

        The panels run from the Leading Edge (LE) to the Trailing Edge
        (TE) of the airfoil camber-line.
        """
        sample_u = self.get_sample_parameters(
            num=self.n_panels + 1, spacing=self.spacing
        )
        camber_points = self.airfoil.camberline_at(sample_u)
        return Panel2D(camber_points)

    @cached_property
    def collocation_points(self):
        return self.panels.points_at(0.25)

    @cached_property
    def unit_rhs_vector(self):
        return self.panels.normals

    @cached_property
    def influence_matrix(self):
        return calc_lumped_vortex_im(
            vortex_pts=self.collocation_points,
            col_pts=self.panels.points_at(0.75),
            panel_normals=self.panels.normals,
        )

    @property
    def solution_class(self) -> Type[ThinFlowSolution]:
        return ThinFlowSolution


@numba.jit(**BASE_NUMBA_CONFIG)
def vortex_2d(  # noqa: D103
    gamma: float, vortex_pt: numba.float64[:, :], col_pt: numba.float64[:, :],
) -> numba.float64[:, :]:
    """Calculates induced velocity due to a vortex at ``vortex_pt``.

    Args:
        gamma: Circulation strength
        vortex_pt: Location of the vortex core
        col_pt: Collocation point to observe the induced velocity

    Returns:
        The induced velocity 2D row-vector at the provided collocation
        point, ``col_pt`` due to vortex of circulation, ``gamma``.
    """

    # Vector from vortex to collocation point
    v_j = col_pt - vortex_pt

    # Squared scalar distance between vortex and collocation point
    r_j2 = v_j[..., 0] ** 2 + v_j[..., 1] ** 2

    # Rotating the vortex to collocation vector 90 degrees CW to obtain
    # the normal vector using a dot product with an Affine Transform
    return (gamma / (2 * math.pi * (r_j2))) * (v_j @ AFFINE_90_CW)


@numba.jit(parallel=True, **BASE_NUMBA_CONFIG)
def calc_lumped_vortex_im(  # noqa: D103
    vortex_pts: numba.float64[:, :],
    col_pts: numba.float64[:, :],
    panel_normals: numba.float64[:, :],
) -> numba.float64[:, :]:
    """Calculates the vortex influence coefficient matrix.

    This is meant to be used with the :py:class:`LumpedVortex`
    panel method which assumes that collocation points and
    vortex points are located at x/c = 0.25 and 0.75 repectively.

    The influence coefficient matrix represents the influence of all
    vortices, index `j`, on all collocation points, index `i`. Initially
    a circulation strength of Gamma = 1 is assumed to obtain the
    geometric influence of a vortex on the current panel. Then the
    induced velocity due to vortex `j` on collocation point `i` is
    calculated with :py:func`vortex_2d` function. The result is then
    projected onto the normal vector of the current panel `i`.

    By projecting this induced velocity onto the panel normal vector, a
    scalar influence coefficient is obtained, which when used with the
    zero normal velocity boundary condition solves for the unknown
    circulation strength of each vortex.

    Args:
        vortex_pts: Vortex points
        col_pts: Collocation points placed along each panel.
        panel_normals: Normal vectors of each panel.

    Returns:
        Vortex influence matrix with assumed circulation, Gamma = 1.
    """
    gamma = 1  # Assuming that the circulation is 1 to solve for
    n_vorts, _ = vortex_pts.shape
    n_cols, _ = col_pts.shape

    influence_matrix = np.zeros((n_cols, n_vorts), dtype=np.float64)

    for i in numba.prange(n_cols):
        n_i = panel_normals[i]
        col_pt = col_pts[i]
        for j in range(n_vorts):
            # Calculating induced velocity at collocation point i due to
            # vortex j, and taking the dot-product with the normal
            # vector to get the influence coefficient a_ij
            influence_matrix[i, j] = (
                vortex_2d(gamma, vortex_pts[j], col_pt) @ n_i
            )

    return influence_matrix


if __name__ == "__main__":
    from gammapy.geometry.airfoil import NACA4Airfoil, ParabolicCamberAirfoil

    a = NACA4Airfoil("naca9512")
    a = ParabolicCamberAirfoil(0.1)

    solver = LumpedVortex(a, 50, "cosine")
    solution = solver.solve_for(alpha=np.arange(5))
    solution.plot_delta_cp(alpha=0)
