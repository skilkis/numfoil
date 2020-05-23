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

"""Contains classes used for interpreting panel method solutions."""

import math
from functools import cached_property
from typing import Optional, Sequence, Union

import numpy as np
from matplotlib import pyplot as plt

from gammapy.geometry.panel import Panel2D


class FlowSolution:
    """Transforms a panel method solution into physical quantitites.

    The output of a panel method are the strengths of the unknown
    singularities placed along the discretized geometry. The
    responsibility of this class is to use this data and output easily
    understood physical quantities such as the lift coefficient and
    pressure coefficient.

    The main benefit of creating this class, other than adhering to the
    Single Responsibility Principle (SRP) is to allow lazy evaluation of
    the physical quantities. This means that if only the
    pressure-coefficient is desired, then the equation for other
    physical quantities, such as the lift coefficient, are not called.

    Args:
        panels: The discretized geometry used to obtain the solution
        circulations: Output singularity strengths from
            :py:class:`PanelMethod`. Assumed to contain N columns
            pertaining to each Angle of Attack specified by ``alpha``.
        density: Density of air in SI kilogram per meter cubed
        velocity: Free-stream velocity in SI meter per second
        alpha: Angle of Attack in SI degree.

    Attributes:
        delta lift:
        delta_pressure:
        delta_pressure_coefficient:
        pressure_coefficient:


    """
    def __init__(
        self,
        panels: Panel2D,
        circulations: np.ndarray,
        density: float,
        velocity: float,
        alpha: Union[float, Sequence[float]],
    ):
        self.panels = panels
        self.circulations = circulations
        self.density = density
        self.velocity = velocity
        self.alpha = alpha

    @cached_property
    def dynamic_pressure(self):
        """Free-stream dynamic pressure in SI Pascal."""
        return 0.5 * self.density * self.velocity ** 2

    @cached_property
    def delta_lift(self):
        """Lift force change across each panel using Kutta-Jukowsi.
        """
        return self.density * self.velocity * self.circulations * self.panels.lengths

    @cached_property
    def delta_pressure(self):
        """Pressure change across each panel using Kutta-Jukowsi."""
        return (
            self.density
            * self.velocity
            * self.circulations
            / self.panels.lengths
        )

    @cached_property
    def delta_pressure_coefficient(self):
        """Pressure coefficient change across each panel."""
        return 2 * self.circulations / (self.panels.lengths * self.velocity)

    @cached_property
    def pressure_coefficient(self):
        """Pressure coefficient measured on each panel."""
        return (
            1
            - (
                self.circulations / (2 * self.velocity)
                - self.panels.tangents @ self.vel_vector.T
            )
            ** 2
        )

    @cached_property
    def lift_coefficient(self):
        """Resultant lift coefficient of the current panel geometry."""
        return np.sum(self.delta_lift, axis=0) / self.dynamic_pressure

    def plot_delta_cp(self, alpha: Optional[float] = None):
        fig, ax = plt.subplots()
        alpha_idx = self.alpha.index(alpha) if alpha is not None else 0
        ax.plot(
            self.panels.points_at(0.5)[:, 0],
            self.delta_pressure_coefficient[:, alpha_idx],
        )
        ax.set_xlabel("Normalized Location Along the Chordline [-]")
        ax.set_ylabel("Pressure Coefficient Difference $\\Delta C_P$")

    def plot_pressure_distribution(self, alpha: Optional[None]):
        raise NotImplementedError

    def plot_lift_gradient(self):
        if self.lift_coefficient.size < 2:
            raise ValueError(
                "A lift gradient plot can only be generated when more "
                "than one Angle of Attack, alpha, is specified"
            )
        fig, ax = plt.subplots()
        ax.plot(
            self.alpha,
            self.lift_coefficient,
            marker="o",
            label="Numerical Solution",
        )
        ax.set_xlabel("Angle of Attack, $\\alpha$ [deg]")
        ax.set_ylabel("Lift Coefficient, $C_l$ [-]")

        alpha_min, alpha_max = min(self.alpha), max(self.alpha)
        ax.plot(
            (alpha_min, alpha_max),
            tuple(
                2 * math.pi * math.radians(a) for a in (alpha_min, alpha_max)
            ),
            label="Thin Airfoil Theory, $C_{l_\\alpha} = 2 \\pi$"
        )
        ax.legend(loc="best")
        return fig, ax
