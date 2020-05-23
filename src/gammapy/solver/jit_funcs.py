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

"""Contains Numba JIT compiled panel method functions."""

import math
from typing import Tuple, Union

import numba
import numpy as np

FAST_MATH_FLAGS = {
    # Refer to https://llvm.org/docs/LangRef.html#fast-math-flags
    "nnan": True,
    "ninf": True,
    "nsz": True,
    "arcp": True,
    "contract": True,
    "afn": True,
    "reassoc": True,
}

BASE_NUMBA_CONFIG = {
    "nopython": True,
    "cache": True,
    "fastmath": FAST_MATH_FLAGS,
}


AFFINE_90_CW = np.array([[0, -1], [1, 0]], dtype=np.float64)


@numba.jit(inline="always", **BASE_NUMBA_CONFIG)
def vortex_2d(
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


@numba.jit(inline="always", **BASE_NUMBA_CONFIG)
def gcs_to_pcs(
    point: numba.float64[:], reference: numba.float64[:], angle: float
) -> numba.float64[:]:
    """Transforms ``point`` from global to panel coordinate system.

    Args:
        point: Point to convert to the panel coordinate system
        reference: Start node of the panel
        angle: Panel angle in SI radian
    """

    x_diff, y_diff = point - reference

    x_p = math.cos(angle) * x_diff + math.sin(angle) * y_diff
    y_p = -math.sin(angle) * x_diff + math.cos(angle) * y_diff

    return np.array((x_p, y_p), dtype=np.float64)


@numba.jit(inline="always", **BASE_NUMBA_CONFIG)
def pcs_to_gcs(
    point: Union[Tuple[float, float], numba.float64[:]], angle: float
) -> numba.float64[:]:
    """Transforms ``point`` from panel to global coordinate system.

    Args:
        point: Point to convert to the global coordinate system
        angle: Panel angle in SI radian
    """

    x_p, y_p = point

    x = math.cos(angle) * x_p - math.sin(angle) * y_p
    y = math.sin(angle) * x_p + math.cos(angle) * y_p

    return np.array((x, y), dtype=np.float64)


@numba.jit(inline="always", **BASE_NUMBA_CONFIG)
def vortex_c_2d(
    gamma: float,
    start_pt: numba.float64[:],
    end_pt: numba.float64[:],
    col_pt: numba.float64[:],
    panel_angle: float,
) -> numba.float64[:, :]:
    """Calculates induced velocity due to a constant-strength vortex.

    Args:
        gamma: Circulation strength per unit length
        start_pt: Start of the vortex panel
        end_pt: End of the vortex panel
        col_pt: Collocation point to observe the induced velocity. This
            should be the midpoint of each panel.
        panel_angle: Angle of the vortex panel in SI radian

    Returns:
        The induced velocity at the provided collocation
        point, ``col_pt`` due to a constant-strength vortex of
        circulation per unit-length, ``gamma``.
    """

    # Transforming collocation and end point to panel coordinates
    col_pt_p = gcs_to_pcs(col_pt, start_pt, panel_angle)
    end_pt_p = gcs_to_pcs(end_pt, start_pt, panel_angle)

    # Vector from start/end points to collocation point
    v_start = col_pt_p
    v_end = col_pt_p - end_pt_p

    # Squared distance between start/end points to collocation point
    r_start2 = v_start[..., 0] ** 2 + v_start[..., 1] ** 2
    r_end2 = v_end[..., 0] ** 2 + v_end[..., 1] ** 2

    # Solving for induced velocities in panel coordinates by integrating
    # the induced velocity due to infinitesimal vortices along the
    # length of the vortex panel (pg. 286 of Kats & Plotkin)
    u_p = (gamma / (2 * math.pi)) * (
        math.atan2(v_end[..., 1], v_end[..., 0])
        - math.atan2(v_start[..., 1], v_start[..., 0])
    )
    v_p = -(gamma / (4 * math.pi)) * math.log(r_start2 / r_end2)

    # Transforming panel coordinates to global coordinates
    return pcs_to_gcs((u_p, v_p), panel_angle)


@numba.jit(parallel=True, **BASE_NUMBA_CONFIG)
def calc_lumped_vortex_im(
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
            # vortex j, and taking the dot-product to get the a_ij
            # v_ij = vortex_2d(gamma, vortex_pts[j], col_pt)
            # a_ij = (v_ij[0] * n_i[0]) + (v_ij[1] * n_i[1])
            influence_matrix[i, j] = (
                vortex_2d(gamma, vortex_pts[j], col_pt) @ n_i
            )

    return influence_matrix


@numba.jit(parallel=True, **BASE_NUMBA_CONFIG)
def calc_constant_vortex_im(
    start_pts: numba.float64[:, :],
    end_pts: numba.float64[:, :],
    col_pts: numba.float64[:, :],
    panel_angles: numba.float64[:, :],
    panel_normals: numba.float64[:, :],
    panel_tangents: numba.float64[:, :],
) -> numba.float64[:, :, :]:
    """Calculates vortex induced velocity influence coefficient matrix.

    This is meant to be used with the :py:class:`ConstantVortex` panel
    method which assumes that each panel has a bound constant strength
    vortex element of unknown circulation per unit length, gamma. The
    collocation points are at the midpoint of each panel.

    The influence coefficient matrix represents the influence of all
    vortices, index `j`, on all collocation points, index `i`. Initially
    a circulation strength of gamma = 1 is assumed to obtain the
    geometric influence of a vortex on the current panel. Then
    the induced velocity due to vortex `j` on collocation point `i` is
    calculated with :py:func`vortex_c_2d` function.

    The result is then projected onto the normal and tangent vectors of
    the current panel `i`. This results in two coefficient matrices
    where, the matrix created with the normal vector is required to
    solve for the unknown circulation strengths using the zero-normal
    velocity boundary condition. On the other hand, the matrix created
    with the tangent vector is necessary to calculate the tangent
    velocity at each panel which is required to calculate the
    pressure coefficient.

    Args:
        start_pts: Start nodes (points) of all panels
        end_pts: End nodes (points) of all panels
        col_pts: Collocation points placed at the midpoint of each panel
        panel_angles: Panel angles in SI radian as a column vector
        panel_normals: Panel normal vectors as a set of row vectors
        panel_tangents: Panel tangent vectors as a set of row vectors

    Returns:
        Vortex influence matrix with assumed circulation, gamma = 1. The
        shape of the influence matrix is (2, n_panels, n_panels) since
        for the Constant Strength Vortex method, both the normal and
        tangent induced velocity influence coefficients are required.
        Therefore, index 0 and 1 of the returned matrix correspond to
        the normal and tangent coefficient matrix respectively.
    """
    gamma = 1  # Assuming that the circulation is 1 to solve for
    n_vorts, _ = start_pts.shape
    n_cols, _ = col_pts.shape

    influence_matrix = np.zeros((2, n_cols, n_vorts), dtype=np.float64)

    for i in numba.prange(n_cols):
        n_i = panel_normals[i]
        t_i = panel_tangents[i]
        col_pt = col_pts[i]
        for j in range(n_vorts):
            # Calculating induced velocity at collocation point i due to
            # vortex j, and taking the dot-product to get the a_ij
            if i == j:
                influence_matrix[..., i, j] = 0.5
            else:
                v_induced = vortex_c_2d(
                    gamma,
                    start_pts[j],
                    end_pts[j],
                    col_pt,
                    panel_angles[j, 0],
                )
                influence_matrix[..., i, j] = v_induced @ n_i, v_induced @ t_i

    return influence_matrix
