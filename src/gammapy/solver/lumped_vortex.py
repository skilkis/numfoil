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

from functools import cached_property

from gammapy.geometry.panel import Panel2D
from gammapy.solver.jit_funcs import calc_lumped_vortex_im
from gammapy.solver.method import PanelMethod


class LumpedVortex(PanelMethod):
    @cached_property
    def unit_rhs_vector(self):
        return self.panels.normals

    @cached_property
    def influence_matrix(self):
        return calc_lumped_vortex_im(
            vortex_pts=self.panels.points_at(0.25),
            col_pts=self.panels.points_at(0.75),
            panel_normals=self.panels.normals,
        )

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

if __name__ == "__main__":
    from gammapy.geometry.airfoil import NACA4Airfoil, ParabolicCamberAirfoil
    from matplotlib import pyplot as plt

    a = NACA4Airfoil("naca9512")
    a = ParabolicCamberAirfoil(0.1)

    Q_inf = 10
    rho = 1.225
    alpha = [1, 2]

    solver = LumpedVortex(a, 200, "linear")
    solution = solver.solve_for(density=1.225, velocity=10, alpha=10)
    plt.plot(-solution.pressure_coefficient)
    solution.plot_delta_cp()
