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

class PanelMethod:

    def __init__(self, Naca=[0,0,0,0], n_panels=5, alpha=1, v_inf=1):
        # * Constants * #
        self.alpha = alpha          # angle of attack
        self.v_inf = v_inf          # freestream velocity
        self.n_panels = n_panels    # Number of panels per surface
        self.NACA = Naca            # NACA XXXX digits of 4-digit airfoil

        self.Q_inf = v_comp(v_inf, alpha)
        self.camberLine = CamberLine(self.NACA, self.n_panels)
        


    # def solve_vorticity(self, surface, Q_inf):


    #     A = np.zeros((n_panels, n_panels))
    #     RHS = np.zeros((n_panels, 1))

    #     for i, col in enumerate(coor_col):
    #         for j, vor in enumerate(coor_vor):
    #             q_ij = v_ind(col[0], col[1], vor[0], vor[1])
    #             a_ij = np.dot(q_ij, normal[i])
    #             A[i][j] = a_ij
    #         RHS[i] = -np.dot(Q_inf, np.transpose(normal[i]))
    #     print(A)
    #     print(RHS)
    #     Gamma = np.linalg.solve(A, RHS)
    #     return Gamma, panel_length, coor_col


    def solve_cp(self):
        dCp = dL = dP = np.zeros((self.n_panels, 1))

        for i in range(len(Gamma)):
            # dL[i] = rho * v_inf * gamma
            # dP[i] = rho * v_inf * Gamma[i] / panel_length[i]
            dCp[i] = -2 * self.Gamma[i] / self.panel_length[i] / self.v_inf

        return dCp


    # self.Gamma, self.panel_length, self.col = self.solve_vorticity(self.coor_c, self.Q_inf)
    # _, _, self.dCp = solve_cp(self.Gamma, self.panel_length)

    def plt(self):
        upper_surface, lower_surface, _ = airfoil(self.NACA, self.n_panels)
        # plt.figure(1)
        plt.plot(
                *zip(*self.camberLine.coor), "k", 
                *zip(*upper_surface), "b", 
                *zip(*lower_surface), "b",
                )
        plt.axis("equal")
        plt.show()


class CamberLine:

    def __init__(self, NACA, n_panels):
        _, _, self.coor = airfoil(NACA, n_panels)
        self.collocation_points, self.vortex_points, self.panel_angles, self.panel_length = panels(self.coor)
        self.normals, self.tangents = normals(self.panel_angles)

    def vorticity(self):
        

    def plt(self):
        # plt.figure(2)
        plt.plot(
                *zip(*self.coor), "k",
                *zip(*self.collocation_points), 'o',
                *zip(*self.vortex_points), 'x'
                )
        plt.show()


class Vorticity:
    def __init__(self, surface, Q_inf):

        self.A = np.zeros((n_panels, n_panels))
        self.RHS = np.zeros((n_panels, 1))

        for i, col in enumerate(coor_col):
            for j, vor in enumerate(coor_vor):
                q_ij = v_ind(col[0], col[1], vor[0], vor[1])
                a_ij = np.dot(q_ij, normal[i])
                A[i][j] = a_ij
            RHS[i] = -np.dot(Q_inf, np.transpose(normal[i]))

        Gamma = np.linalg.solve(A, RHS)
        return Gamma, panel_length, coor_col