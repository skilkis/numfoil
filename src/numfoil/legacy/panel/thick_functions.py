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

"""Contains useful geometric functions for airfoils and vectors."""

from math import atan, cos, pi, sin, sqrt, atan2, log
from typing import List, Tuple

import numpy as np


def v_comp(v_inf: float, alpha: float) -> np.ndarray:
    """Returns a velocity component vector.

    Args:
        v_inf: Freestrean velocity
        alpha: Angle of attack in degrees

    Returns:
        A velocity vector
    """
    AoA = alpha  # /* pi / 180
    u_inf = v_inf * cos(AoA)
    w_inf = v_inf * sin(AoA)
    Q_inf = np.array([[u_inf, w_inf]])
    return Q_inf


# TODO this violates SRP, consider using a class to generate camber-line
# when required

# TODO  deprecate after transitioning to using a class here
# @deprecated(version="0.1.0", reason="Use the new `naca_airfoil()` function")
def airfoil(Naca=[2, 4, 1, 0], n_panels=10):
    """Creates a NACA 4-series airfoil given by ``Naca``.

    This function finds the function of the camber-line,
    thickness distribution, and upper and lower surface points using
    the supplied ``Naca``

    Args:
        Naca (list, optional): list containing 4 digits of a NACA
        airfoil. Defaults to [2, 4, 1, 0].
        n_panels (int, optional): Number of panels.
                           Defaults to 10.

    Returns:
        coor_u (list): List of coordinates of the upper airfoil surface.
        coor_l (list): List of coordinates of the lower airfoil surface.
        coor_c (list): List of coordinates of points on the camber line.
    """
    M = Naca[0] / 100
    P = Naca[1] / 10
    t_c = float("0.{}{}".format(Naca[2], Naca[3]))

    N = int(n_panels/2 +1)  # Number of panel edges

    a0 = 0.2969
    a1 = -0.126
    a2 = -0.3516
    a3 = 0.2843
    a4 = -0.1036  # -0.1036 for closed TE, -0.1015 for open TE

    coor_u = []
    coor_l = []
    coor_c = []
    coor_t = []

    # using NACA 4 digit airfoil calculation to
    # obtain airfoil coordinate points = panel edges
    # ! -------------------------------------------
    for beta in np.linspace(0, pi, N):
        # cosine spacing
        x = 0.5 - 0.5 * cos(beta)
        # ! -------------------------------------------
    # for x in np.linspace(0, 1, N):
        # ! -------------------------------------------
        # thickness distribution
        y_t = (
            t_c
            / 0.2
            * (
                a0 * x ** 0.5
                + a1 * x
                + a2 * x ** 2
                + a3 * x ** 3
                + a4 * x ** 4
            )
        )

        if x < P:
            # camber line before t_max
            y_c = M / P ** 2 * (2 * P * x - x ** 2)
            # gradient before t_max
            dyc_dx = 2 * M / P ** 2 * (P - x)

        elif x >= P:
            # camber line after t_max
            y_c = M / (1 - P) ** 2 * (1 - 2 * P + 2 * P * x - x ** 2)
            # gradient after t_max
            dyc_dx = 2 * M / (1 - P) ** 2 * (P - x)

        theta = atan(dyc_dx)

        # exact position of upper and lower surface coordinates
        # perpendicular to camber line
        x_u = x - y_t * sin(theta)
        y_u = y_c + y_t * cos(theta)
        x_l = x + y_t * sin(theta)
        y_l = y_c - y_t * cos(theta)

        coor_u.append([x_u, y_u])  # upper airfoil coordinates
        coor_l.append([x_l, y_l])  # lower airfoil coordinates
        coor_t.append([x, y_t])  # thickness distribution
        coor_c.append([x, y_c])  # camber line coordinates = panel edges

    return coor_u, coor_l, coor_c


def make_panels(coor: list) -> Tuple[list, list, list, list]:
    """Creates panels from a list of points on a camber-line.

    This function takes a list of points on the camber line provided by
    `coor`. These points are considered panel edges. Collocation points
    are added at 0.75 panel length and vortex points are added at 0.25
    panel lenght.

    Args:
        coor: List of panel edges (on camber line).
                       should be list of camberline points.

    Returns:
        coor_col: list of coordinates of collocation points.
        coor_vor: list of coordinates of vortex element points.
        thetas: list of panel angles.
        panel_lengths: list of panel lengths.
    """
    coor_col = []
    coor_vor = []
    thetas = []
    panel_lengths = []

    for i in range(len(coor) - 1):
        p1 = coor[i]
        p2 = coor[i + 1]

        # place collocation point at 0.75 panel length
        x_col = p1[0] + 0.5 * (p2[0] - p1[0])  # x_i
        y_col = p1[1] + 0.5 * (p2[1] - p1[1])  # y_i
        coor_col.append([x_col, y_col])

        # place vortex point at 0.25 panel length
        x_vor = p1[0] + 0 * (p2[0] - p1[0])  # x_j
        y_vor = p1[1] + 0 * (p2[1] - p1[1])  # x_j
        coor_vor.append([x_vor, y_vor])

        # find panel angle
        theta = atan2((p2[1] - p1[1]), (p2[0] - p1[0]))
        thetas.append(theta)

        # panel length
        length = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        panel_lengths.append(length)

    return coor_col, coor_vor, thetas, panel_lengths


# TODO vectorize this function using either numpy or numba
def v_ind_loc(
    x_col: float, y_col: float, x1_panel: float, y1_panel: float, x2_panel: float, y2_panel: float, gamma: float = 1
) -> np.ndarray:
    """Finds induced velocity at a point ``x_col``, ``y_col``.

    The induced velocity is due to a vortex with circulation ``Gamma``
    at ``x_vor``, ``y_vor``.

    Args:
        x_col: x location of evaluation point
        y_col: y location of evaluation point
        x_vor: x location of vortex element
        y_vor: y location of vortex element
        Gamma: Vorticity. Defaults to unit strength 1.

    Returns:
        u: horizontal component of induced velocity due to vortex
        y: vertical component of induced velocity due to vortex
    """

    u_p = gamma / ( 2 * pi) * ( atan2( (y_col - y2_panel),  (x_col - x2_panel) ) 
                              - atan2( (y_col - y1_panel), (x_col - x1_panel) ) )
    w_p = -gamma / ( 4 * pi) * log( 
        ( (x_col - x1_panel)**2 + (y_col - y1_panel)**2 )/( (x_col - x2_panel)**2 + (y_col - y2_panel)**2) 
        )

    return np.array([u_p, w_p])


def normals(thetas: list) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Returns normal and tangent vectors for a list of panel angles.

    Args:
        thetas: List of panel angels

    Returns:
        normals (list): list of numpy arrays containing the normal
            vectors for all panels
        tangents (list): list of numpy arrays
            containing the tangent vectors for all panels
    """
    normals = []
    tangents = []
    for theta in thetas:
        n_i = np.array([sin(theta), cos(theta)])  # normal vector
        t_i = np.array([cos(theta), -sin(theta)])  # tangent vector
        normals.append(n_i)
        tangents.append(t_i)
    return np.array(normals), np.array(tangents)
