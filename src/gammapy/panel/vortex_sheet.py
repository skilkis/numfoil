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
from math import pi, sin
from gammapy.functions import airfoil, normals, panels, v_comp, v_ind

# pg280 573 300

class PanelledAirfoil:

    def __init__(self, Naca=[0,0,0,0], n_panels=5, alpha=1, v_inf=1):
        self.alpha = alpha *pi/180  # angle of attack
        self.v_inf = v_inf          # freestream velocity
        self.n_panels = n_panels    # Number of panels per surface
        self.NACA = Naca            # NACA XXXX digits of 4-digit airfoil

        self.Q_inf = v_comp(v_inf, alpha)
        self.camberline = CamberLine(self.NACA, self.n_panels)
        # setattr(self.camberline, 'vorticity', Vorticity(self.camberline, v_inf, alpha))
        setattr(self.camberline, 'vorticity', Vorticity(self.camberline, self.Q_inf))

    def solve_cp(self, plot=False):
        dCp = dL = dP = np.zeros((self.camberline.n_panels, 1))
        for i in range(len(self.camberline.vorticity.Gamma)):
            # dL[i] = rho * v_inf * gamma
            # dP[i] = rho * v_inf * Gamma[i] / panel_lengths[i]
            dCp[i] = -2 * self.camberline.vorticity.Gamma[i] / self.camberline.panel_lengths[i] / self.v_inf
        
        if plot is True:
            x = [i[0] for i in self.camberline.collocation_points]
            plt.plot(x, dCp)
            plt.show()

        return dCp

    def plt(self):
        upper_surface, lower_surface, _ = airfoil(self.NACA, self.n_panels)
        plt.plot(
                *zip(*self.camberline.coor), "k", 
                *zip(*upper_surface), "b", 
                *zip(*lower_surface), "b",
                )
        plt.axis("equal")
        plt.show()


class CamberLine:

    def __init__(self, NACA, n_panels):
        self.n_panels = n_panels
        _, _, self.coor = airfoil(NACA, self.n_panels)
        self.collocation_points, self.vortex_points, self.panel_angles, self.panel_lengths = panels(self.coor)
        self.normals, self.tangents = normals(self.panel_angles)

    def plt(self):
        plt.plot(
                *zip(*self.coor), "k",
                *zip(*self.collocation_points), 'o',
                *zip(*self.vortex_points), 'x'
                )
        plt.show()


class Vorticity:

    # def __init__(self, camberline, v_inf, alpha):
    def __init__(self, camberline, Q_inf):
        self.A = np.zeros((camberline.n_panels, camberline.n_panels))
        self.RHS = np.zeros((camberline.n_panels, 1))

        for i, col in enumerate(camberline.collocation_points):
            for j, vor in enumerate(camberline.vortex_points):
                q_ij = v_ind(col[0], col[1], vor[0], vor[1])
                a_ij = np.dot(q_ij, camberline.normals[i])
                self.A[i][j] = a_ij
            self.RHS[i] = -np.dot(Q_inf, np.transpose(camberline.normals[i]))
            # self.RHS[i] = -v_inf * sin(alpha + camberline.panel_angles[i])


        self.Gamma = np.linalg.solve(self.A, self.RHS)
    
