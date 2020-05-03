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

"""Contains a thin-airfoil theory Vortex Sheet solution."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from deprecated.sphinx import deprecated

from gammapy.functions import airfoil, normals, panels, v_comp, v_ind

# pg280 573 300

# * Constants * #
alpha = 1  # angle of attack
v_inf = 1  # freestream velocity
n_panels = 100  # Number of panels per surface
NACA = [0, 0, 0, 6]  # NACA XXXX digits of 4-digit airfoil


Q_inf = v_comp(v_inf, alpha)
coor_u, coor_l, coor_c = airfoil(NACA, n_panels)


def solve_vorticity(surface):
    coor_col, coor_vor, panel_angle, panel_length = panels(surface)
    normal, tangent = normals(panel_angle)

    A = np.zeros((n_panels, n_panels))
    RHS = np.zeros((n_panels, 1))

    for i, col in enumerate(coor_col):
        for j, vor in enumerate(coor_vor):
            q_ij = v_ind(col[0], col[1], vor[0], vor[1])
            a_ij = np.dot(q_ij, normal[i])
            A[i][j] = a_ij
        RHS[i] = -np.dot(Q_inf, np.transpose(normal[i]))
    Gamma = np.linalg.solve(A, RHS)
    return Gamma, panel_length, coor_col


def solve_cp(Gamma, panel_length):
    dCp = dL = dP = np.zeros((n_panels, 1))

    for i in range(len(Gamma)):
        # dL[i] = rho * v_inf * gamma
        # dP[i] = rho * v_inf * Gamma[i] / panel_length[i]
        dCp[i] = -2 * Gamma[i] / panel_length[i] / v_inf

    return dL, dP, dCp


Gamma_u, panel_length_u, col_u = solve_vorticity(coor_u)
Gamma_l, panel_length_l, col_l = solve_vorticity(coor_l)

_, dP_u, Cp_u = solve_cp(Gamma_u, panel_length_u)
_, dP_l, Cp_l = solve_cp(Gamma_l, panel_length_l)


# * plotting * #


@deprecated(version="0.1.0", reason="Use `zip` from Python STD Lib instead")
def pt(set: List[list], i: int) -> list:
    """Get all x or y-values from a coordinate set for easier plotting.

    i = 0 for x values, i = 1 for y values.

    Args:
        set (list): list of coordinates
        i (integer): Set i=0 for a list
        of x values, set i=1 for a list of all y vlaues

    Returns:
        List of all x- or all y-values from a list of coordinates
    """
    p = []
    for point in set:
        p.append(point[i])
    return p


plt.figure(1)
plt.plot(
    *zip(*coor_c), "k", *zip(*coor_u), "b", *zip(*coor_l), "b",
)
plt.axis("equal")
plt.show()

plt.figure(2)
x_u = [i[0] for i in col_u]
x_l = [i[0] for i in col_l]
plt.plot(x_u, -Cp_u, "b", x_l, Cp_l, "g")
plt.show()
