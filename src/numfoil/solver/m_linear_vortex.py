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

"""Contains definitions for a Linear Strength Vortex Panel Method."""

import math
from functools import cached_property
from typing import Dict, Sequence, Tuple, Union

import numba
import numpy as np

from numfoil.geometry.panel import Panel2D
from numfoil.solver.base import (
    BASE_NUMBA_CONFIG,
    PanelMethod,
    ThickFlowSolution,
)


class LinearVortex(PanelMethod):
    """Implements a Linear Strength Vortex panel method."""

    @cached_property
    def panels(self) -> Panel2D:
        """Panels that run from TE -> Bottom -> Top -> TE."""
        sample_u = PanelMethod.get_sample_parameters(
            num=(self.n_panels // 2) + 1, spacing=self.spacing
        )
        bot_pts = self.airfoil.lower_surface_at(sample_u[::-1])
        top_pts = self.airfoil.upper_surface_at(sample_u[1:])
        # return Panel2D(np.vstack((bot_pts, top_pts, ((1.25, 0)))))
        return Panel2D(np.vstack((bot_pts, top_pts)))

    @cached_property
    def collocation_points(self) -> np.ndarray:
        """Collocation points located at the midpoint of each panel."""
        return self.panels.points_at(0.5)

    @cached_property
    def unit_rhs_vector(self) -> np.ndarray:
        """Normalized right-hand-side consisting of panel normals."""
        # Final entry is (0, 0) which will enforce the Kutta condition
        normals = np.zeros((self.panels.n_panels + 1, 2), dtype=np.float64)
        normals[:-1, :] = self.panels.normals
        return normals

    @cached_property
    def influence_matrices(self) -> Dict[str, np.ndarray]:
        """Normal and tangent influence coefficient matrices.

        Note:
            As the tangent influence matrix is only used for post
            processing with :py:class:`ThickFlowSolution` the final
            entry which represents the Kutta-condition does not need to
            be included.
        """
        start_pts, _ = self.panels.nodes
        im_normal, im_tangent = calc_linear_vortex_im(
            col_pts=self.collocation_points,
            vort_pts=start_pts,  # Vortices are placed on all start nodes
            panel_angles=self.panels.angles,
            panel_lengths=self.panels.lengths,
        )
        return {"normal": im_normal, "tangent": im_tangent[:-1, :-1]}

    @property
    def influence_matrix(self) -> np.ndarray:
        """Normal influence matrix for :py:meth`get_circulations`."""
        return self.influence_matrices["normal"]

    def solve_for(
        self, alpha: Union[float, Sequence[float]], plot: bool = False,
    ) -> ThickFlowSolution:
        """Returns a :py:class:`FlowSolution` with lazy attributes.

        Args:
            alpha: A value or sequence of Angle of Attack in SI degree
            plot: Sets if plots should be shown on initialization.
                This evaluates all attributes and negates the
                laziness of the returned object. Defaults to 1.
        """
        return self.solution_class(
            method=self,
            # Removing constant error term in the solution
            circulations=self.get_circulations(alpha)[:-1, :],
            alpha=alpha,
        )

    @property
    def solution_class(self):
        return ThickFlowSolution


@numba.jit(**BASE_NUMBA_CONFIG)
def calc_integration_constants(  # noqa: D103
    col_pt: numba.float64[:],
    vort_pt: numba.float64[:],
    col_panel_angle: float,
    vort_panel_angle: float,
    vort_panel_length: float,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Calculates integration constants of the linear vortex method.

    Refer to pg. 157-158 of Foundations of Aerodynamics: 5th Edition or
    watch JoshTheEngineer's `video_`.

    Args:
        col_pt: Collocation point to measure induced velocity
        vort_pt: Start node (point) of a linear vortex panel
        col_panel_angle: Panel angle of the collocation point panel
        vort_panel_angle: Panel angle of the linear vortex panel
        vort_panel_length: Length of the linear vortex panel

    Returns:
        A, B, C, D, E, F, G, P, Q integration constants of the
        linear vortex panel method for airfoils of arbitrary thickness.

    .. _video: https://www.youtube.com/watch?v=5lmIv2CUpoc
    """

    v = col_pt - vort_pt  # Vector from vortex to collocation point
    vx, vy = v  # Vector from vortex to collocation point
    s_j = vort_panel_length  # Length of the vortex panel

    a = -vx * math.cos(vort_panel_angle) - vy * math.sin(vort_panel_angle)
    b = np.sum(v ** 2)
    c = math.sin(col_panel_angle - vort_panel_angle)
    d = math.cos(col_panel_angle - vort_panel_angle)
    e = vx * math.sin(vort_panel_angle) - vy * math.cos(vort_panel_angle)
    f = math.log(1 + ((s_j ** 2 + 2 * s_j * a) / b))
    g = math.atan2(e * s_j, b + a * s_j)

    # Turning Black formatter off here for redability
    # fmt: off
    p = vx * math.sin(col_panel_angle - 2 * vort_panel_angle) \
        + vy * math.cos(col_panel_angle - 2 * vort_panel_angle)
    q = vx * math.cos(col_panel_angle - 2 * vort_panel_angle) \
        - vy * math.sin(col_panel_angle - 2 * vort_panel_angle)
    # fmt: on

    return a, b, c, d, e, f, g, p, q


@numba.jit(**BASE_NUMBA_CONFIG)
def calc_vortex_coefficients(  # noqa: D103
    col_pt: numba.float64[:],
    vort_pt: numba.float64[:],
    col_panel_angle: float,
    vort_panel_angle: float,
    vort_panel_length: float,
) -> Tuple[float, float]:
    """Calculates induced velocity coefficients.

    Refer to pg. 157-159 of Foundations of Aerodynamics: 5th Edition.

    Args:
        col_pt: Collocation point to measure induced velocity
        vort_pt: Start node (point) of a linear vortex panel
        col_panel_angle: Panel angle of the collocation point panel
        vort_panel_angle: Panel angle of the linear vortex panel
        vort_panel_length: Length of the linear vortex panel

    Returns:
        CN_1, CN_2, CT_1, CT_2 induced velocity coefficients. The
        letter N and T refer to the normal and tangential induced
        velocities respectively. Furthermore, the subscript 1 is the
        velocity starting intensity
    """
    a, b, c, d, e, f, g, p, q = calc_integration_constants(
        col_pt, vort_pt, col_panel_angle, vort_panel_angle, vort_panel_length,
    )
    s_j = vort_panel_length

    cn_2 = d + (0.5 * q * f / s_j) - (a * c + d * e) * g / s_j
    cn_1 = (0.5 * d * f) + (c * g) - cn_2

    ct_2 = c + (0.5 * p * f / s_j) + (a * d - c * e) * g / s_j
    ct_1 = (0.5 * c * f) - (d * g) - ct_2

    return cn_1, cn_2, ct_1, ct_2


@numba.jit(parallel=True, **BASE_NUMBA_CONFIG)
def calc_linear_vortex_im(  # noqa: D103
    col_pts: numba.float64[:, :],
    vort_pts: numba.float64[:, :],
    panel_angles: numba.float64[:, :],
    panel_lengths: numba.float64[:, :],
) -> numba.float64[:, :, :]:
    """Calculates the normal and tangent influence coefficient matrices.

    This is meant to be used with the :py:class:`LinearVortex` panel
    method which assumes that there are n+1 vortices of unknown
    circulation per unit length gamma. Therefore, the Kutta-condition
    can be satisfied without deleting or adding any additional
    equations.

    Args:
        start_pts: Start nodes (points) of all panels
        end_pts: End nodes (points) of all panels
        col_pts: Collocation points placed at the midpoint of each panel
        panel_angles: Panel angles in SI radian as a column vector
        panel_normals: Panel normal vectors as a set of row vectors
        panel_tangents: Panel tangent vectors as a set of row vectors

    Returns:
        Normal and tangent vortex influence matrices for the Linear
        Strength Vortex method returned as a 3D numpy array. The shape
        of the returned array is (2, n_panels + 1, n_panels +1) thus the
        normal and tangent vortex influence matrics can be unpacked as
        follows::

            im_normal, im_tangent = calc_linear_vortex_im(*args)
    """
    n_vorts, _ = vort_pts.shape
    n_cols, _ = col_pts.shape

    influence_matrix = np.zeros((2, n_cols + 1, n_vorts + 1), dtype=np.float64)

    # cn = Normal induced velocity coefficient
    # ct = Tangent induced velocity coefficient
    # Subscript 1 = Panel start value
    # Subscript 2 = Panel end value
    for i in numba.prange(n_cols):

        col_pt = col_pts[i]
        col_panel_angle = panel_angles[i, 0]

        # Initial end value coefficients are zero
        cn_2_old, ct_2_old = float(0), float(0)
        for j in range(n_vorts):
            if i == j:
                cn_1, cn_2, ct_1, ct_2 = -1, 1, math.pi / 2, math.pi / 2
            else:
                cn_1, cn_2, ct_1, ct_2 = calc_vortex_coefficients(
                    col_pt=col_pt,
                    vort_pt=vort_pts[j],
                    col_panel_angle=col_panel_angle,
                    vort_panel_angle=panel_angles[j, 0],
                    vort_panel_length=panel_lengths[j, 0],
                )
            # Storing normal/tangent coefficient and updating end value
            influence_matrix[..., i, j] = cn_1 + cn_2_old, ct_1 + ct_2_old
            cn_2_old, ct_2_old = cn_2, ct_2

        # Vortex at TE gets final end value coefficients
        influence_matrix[..., i, j + 1] = cn_2_old, ct_2_old

    # Inserting kutta condition gamma_0 + gamma_n+1 = 0 for normal
    # coefficient matrix (Index = 0)
    influence_matrix[0, -1, 0] = 1
    influence_matrix[0, -1, -1] = 1

    return influence_matrix
