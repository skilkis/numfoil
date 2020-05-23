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

"""Implements a First Order Constant Strength Vortex Panel Method."""

from functools import cached_property
from typing import Dict, Sequence, Union

import numpy as np

from gammapy.geometry.panel import Panel2D
from gammapy.solver.jit_funcs import calc_constant_vortex_im
from gammapy.solver.method import PanelMethod

# Great source explaining it
# https://www.youtube.com/watch?v=Ai0o5ppUTuk

# Validation w/ Analytical Solution
# https://www.youtube.com/watch?v=aAmr6YpbwaQ

# Panel Coordinates Transformation
# https://www.youtube.com/watch?v=i8nyy9_TGOY


class ConstantVortex(PanelMethod):
    @cached_property
    def unit_rhs_vector(self):
        # Final entry is (0, 0) which will enforce the Kutta condition
        normals = np.zeros((self.panels.n_panels + 1, 2), dtype=np.float64)
        normals[:-1, :] = self.panels.normals
        return normals

    @cached_property
    def panels(self) -> Panel2D:
        """Returns panels that run from TE -> Bottom -> Top -> TE."""
        sample_u = PanelMethod.get_sample_parameters(
            num=(self.n_panels // 2) + 1, spacing=self.spacing
        )
        bot_pts = self.airfoil.lower_surface_at(sample_u[::-1])
        top_pts = self.airfoil.upper_surface_at(sample_u[1:])
        # return Panel2D(np.vstack((bot_pts, top_pts, ((1.25, 0)))))
        return Panel2D(np.vstack((bot_pts, top_pts)))

    @cached_property
    def influence_matrices(self) -> Dict[str, np.ndarray]:
        start_pts, end_pts = self.panels.nodes
        im_normal, im_tangent = calc_constant_vortex_im(
            start_pts=start_pts,
            end_pts=end_pts,
            col_pts=self.panels.points_at(0.5),
            panel_angles=self.panels.angles,
            panel_normals=self.panels.normals,
            panel_tangents=self.panels.tangents,
        )
        return {"normal": im_normal, "tangent": im_tangent}

    @cached_property
    def influence_matrix(self) -> np.ndarray:
        n_panels = self.panels.n_panels
        im_normal = self.influence_matrices["normal"]

        # Setting final row to kutta condition, summation of circulation
        # of first and last panel = 0
        kutta_row = np.zeros((n_panels + 1), dtype=np.float64)
        kutta_row[[0, -2]] = 1

        # Creating an unknown constant error, e, to make the
        # overconstrained N+1 system solveable (Moran pg. 282, 1984)
        error_column = np.ones((n_panels + 1), dtype=np.float64)

        # Creating an empty N+1, M+1 influence matrix and broadcasting
        # Kutta condition row as well as the constant error column
        solveable_im = np.zeros((n_panels + 1, n_panels + 1), dtype=np.float64)
        solveable_im[:-1, :-1] = im_normal
        solveable_im[:, -1] = error_column

        # Broadcasting Kutta row shouldn't contain an error term
        solveable_im[-1, :] = kutta_row

        return solveable_im

    def solve_for(
        self,
        density: float,
        velocity: float,
        alpha: Union[float, Sequence[float]],
        plot: bool = False,
    ) -> object:
        """."""
        return self.__solution_class__(
            panels=self.panels,
            circulations=self.get_circulation_strengths(velocity, alpha)[
                :-1, :
            ],
            density=density,
            velocity=velocity,
            alpha=alpha,
        )


if __name__ == "__main__":
    from gammapy.geometry.airfoil import NACA4Airfoil
    from matplotlib import pyplot as plt

    a = NACA4Airfoil("naca0012", te_closed=True)

    alpha = 8
    velocity = 20
    panels = 200

    solver = ConstantVortex(a, panels, "cosine")
    solution = solver.solve_for(density=1.225, velocity=velocity, alpha=alpha)

    cp = (
        1
        - (
            (solver.influence_matrices["tangent"] / velocity)
            @ (solution.circulations)
            + (np.cos(solver.panels.angles - np.radians(alpha)))
        )
        ** 2
    )
    stag_idx = np.argwhere(cp == np.max(cp))[0, 0]
    plt.plot(
        solver.panels.points_at(0.5)[:stag_idx, 0],
        cp[:stag_idx],
        label="Lower Surface",
    )
    plt.plot(
        solver.panels.points_at(0.5)[stag_idx - 1 :, 0],
        cp[stag_idx - 1 :],
        label="Upper Surface",
    )
    plt.scatter(
        solver.panels.points_at(0.5)[stag_idx, 0],
        cp[stag_idx],
        marker="o",
        label="Stagnation Point",
        zorder=3,
        color="black",
        facecolor="white",
    )
    plt.legend(loc="best")
    plt.gca().invert_yaxis()
