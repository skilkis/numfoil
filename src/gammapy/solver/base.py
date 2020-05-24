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

"""Contains definitions used to define a panel method solver."""

import math
from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import Optional, Sequence, Type, Union

import numpy as np
from matplotlib import pyplot as plt

from gammapy.geometry import Airfoil
from gammapy.geometry.panel import Panel2D

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
        method: The :py:class:`PanelMethod` used to obtain circulations
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
        method: object,
        circulations: np.ndarray,
        alpha: Union[float, Sequence[float]],
    ):
        self.method = method
        self.circulations = circulations
        self.alpha = alpha

    # @cached_property
    # def delta_lift(self):
    #     """Lift force change across each panel using Kutta-Jukowsi.
    #     """
    #     return self.circulations * self.panels.lengths

    # @cached_property
    # def delta_pressure(self):
    #     """Pressure change across each panel using Kutta-Jukowsi."""
    #     return (
    #         self.density
    #         * self.velocity
    #         * self.circulations
    #         / self.panels.lengths
    #     )

    @cached_property
    def delta_pressure_coefficient(self):
        """Pressure coefficient change across each panel."""
        return 2 * self.circulations / self.method.panels.lengths

    @cached_property
    def pressure_coefficient(self):
        """Pressure coefficient measured on each panel."""
        return (
            1
            - (
                self.circulations / (2 * self.velocity)
                - self.method.panels.tangents @ self.vel_vector.T
            )
            ** 2
        )

    @cached_property
    def lift_coefficient(self) -> float:
        """Resultant lift coefficient of the current panel geometry."""
        return np.sum(2 * self.circulations, axis=0)

    def plot_delta_cp(self, alpha: Optional[float] = None):
        fig, ax = plt.subplots()
        alpha_array = np.array(self.alpha)
        alpha_idx = (alpha_array == alpha) if alpha is not None else ...
        ax.plot(
            self.method.panels.points_at(0.5)[:, 0],
            self.delta_pressure_coefficient[:, alpha_idx],
            marker="o",
            markeredgecolor="black",
            markerfacecolor="white",
        )
        ax.legend(
            loc="best",
            # Uses alpha_idx to label each of the pressure dists. If
            # ``alpha`` is None then the ellipsis operator will index
            # all elements in the Numpy array of alphas
            labels=[f"$\\alpha = {a}$" for a in alpha_array[alpha_idx]],
        )
        ax.set_xlabel("Normalized Location Along the Chordline [-]")
        ax.set_ylabel("Pressure Coefficient Difference $\\Delta C_P$")

    def plot_pressure_distribution(self, alpha: Optional[None]):
        raise NotImplementedError

    def plot_lift_gradient(self, label: Optional[str] = "Numerical Solution"):
        if self.lift_coefficient.size < 2:
            raise ValueError(
                "A lift gradient plot can only be generated when more "
                "than one Angle of Attack, alpha, is specified"
            )
        fig, ax = plt.subplots()
        ax.plot(
            self.alpha, self.lift_coefficient, marker="o", label=label,
        )
        ax.set_xlabel("Angle of Attack, $\\alpha$ [deg]")
        ax.set_ylabel("Lift Coefficient, $C_l$ [-]")

        alpha_min, alpha_max = min(self.alpha), max(self.alpha)
        ax.plot(
            (alpha_min, alpha_max),
            tuple(
                2 * math.pi * math.radians(a) for a in (alpha_min, alpha_max)
            ),
            label="Thin Airfoil Theory, $C_{l_\\alpha} = 2 \\pi$",
        )
        ax.legend(loc="best")
        return fig, ax


class PanelMethod(metaclass=ABCMeta):
    """Defines an Abstract Base Class (ABC) for all panel methods.

    Note:
        When overriding the :py:funtion`abc.abstractmethod` decorated
        properties use of :py:function:`functools.cached_property`
        should be preferred so as to only perform calculations once.

    Args:
        airfoil: An instance of :py:class:`Airfoil`
        n_points: Number of points to sample on the chord-line.
        spacing: Sets the spacing used for the points on the
            chord-line between 0-1. Available options are "cosine"
            and "linear". A linear spacing will return an array of
            points that are equidistant from each other. Whereas a
            cosine spacing increases accuracy of some aerodynamic
            solvers by increasing the density of points close to the
            leading and trailing edges. Defaults to "cosine".
    """

    def __init__(
        self,
        airfoil: Airfoil,
        n_panels: int,
        spacing: Optional[str] = "cosine",
    ):
        # Setting attributes with object.__setattr__ since
        # PanelMethod.__setattr__ is blocked for these attributes
        super().__setattr__("airfoil", airfoil)
        super().__setattr__("n_panels", n_panels)
        super().__setattr__("spacing", spacing)

    def __setattr__(self, name, value):
        """Makes initialization arguments unsettable."""
        init_args = ("airfoil", "n_panels", "spacing")
        if name in init_args:
            raise AttributeError(
                "Input arguments to a PanelMethod cannot be changed, "
                "please create a new instance with the new inputs"
            )
        super().__setattr__(name, value)

    @property
    @abstractmethod
    def panels(self) -> Panel2D:
        """Returns the panelled representation of :py:attr:`airfoil."""
        ...

    @property
    @abstractmethod
    def collocation_points(self) -> np.ndarray:
        """Returns the collocation points used in the solution."""
        ...

    @property
    @abstractmethod
    def influence_matrix(self) -> np.ndarray:
        """Returns a matrix of influence coefficients.

        Each element in the matrix describes the induced velocity
        measured at a collocation point due to a single singularity of
        unit-strength. A single row of the matrix is then all induced
        velocities measured at a collocation point, index i, due to
        all vortices in the system, index j = 1 to N. The influence
        coefficients purely describe the geometric coupling of the
        panels and is independent of the flow condition.
        """
        ...

    @property
    @abstractmethod
    def unit_rhs_vector(self) -> np.ndarray:
        """."""
        ...

    @property
    def solution_class(self) -> Type[FlowSolution]:
        """Flow solution class used to obtain physical quantities.

        This can be overridden in specializations to change the behavior
        of how physical quantities are calculated using the circulation
        strengths obtained by the :py:class:`PanelMethod`.
        """
        return FlowSolution

    # TODO a solution should have cp, cl, delta cp THATS IT
    def solve_for(
        self, alpha: Union[float, Sequence[float]], plot: bool = False,
    ) -> FlowSolution:
        """."""
        return self.solution_class(
            method=self,
            circulations=self.get_circulations(alpha),
            alpha=alpha,
        )

    def get_circulations(
        self, alpha: Union[float, Sequence[float]]
    ) -> np.ndarray:
        """Solves the linear system to obtain circulation strengths.

        To obtain the circulation strengths, `x`, the linear system `Ax
        = b` must be solved. Here, `A` is :py:attr:`influence_matrix`
        which is dependent only on the panel geometry. On the other
        hand, `b` is the boundary conditions of the system, referred to
        as the Right Hand Side (RHS), which are derived from the zero
        penetration velocity conditions and is dependent on both the
        panel geometry and the flow angle and velocity. As a result,
        although the `A` matrix can be evaluated once for all velocities
        and Angle of Attacks (AoAs), the `b` RHS vector must be
        recalculated for each flow condition.

        Depending on the specific :py:class:`PanelMethod`
        specialization, the boundary conditions can either involve the
        normal or tangent vectors of each panel. To account for this,
        :py:attr:`unit_rhs_vector` is calculated once, and for each
        velocity and AoA specification, the resultant boundary condition
        is obtained by taking the dot product with the local velocity
        vector. In essence, this projects the velocity vector onto the
        :py:attr:`unit_rhs_vector` which provides the correct boundary
        condition for each :py:class:`PanelMethod`.

        Args:
            alpha: Angle of Attack (AoA) in SI degree. If a sequence of
                N AoAs are specified then the solver will solve the
                linear system, N times and return the resultant
                circulations in a 2D Numpy array where the rows
                contain the circulation strength at each panel and
                the columns pertain to the respective N AoAs.

        Returns:
            Circulation strengths for all singularities specified by
            the discrete :py:class:`PanelMethod` for the flow
            condition(s) specified by ``velocity`` and ``alpha``. If
            ``alpha`` is a sequence, then the columns of the returned
            array will represent the circulation strengths at each
            Angle of Attack (AoA).
        """

        a_matrix = self.influence_matrix
        flow_dir = self.get_flow_direction(alpha)

        # Obtaining the Right-Hand-Side RHS by taking the dot product
        # with the flow direction. Resultant dimensions of the RHS
        # vector is (N_panels, 2), (2, N_alpha) -> (N_panels, N_alpha)
        rhs = self.unit_rhs_vector @ -flow_dir.T

        # Solution of the system has dimensions (N_panels, N_alpha)
        return np.linalg.solve(a_matrix, rhs)

    # TODO change to get_flow_direction
    @staticmethod
    def get_flow_direction(alpha: Union[float, Sequence[float]]) -> np.ndarray:
        """Convert ``alpha`` into a flow direction vector.

        If an iterable sequence of N Angle of Attacks (AoAs) are given
        then the return statement will be a set of 2D row-vectors with
        shape (N, 2).

        Args:
            alpha: Airfoil Angle of Attack (AoA) in SI degree
        """
        # Converting alpha to radians and ensuring that it is always 1D
        alpha_rad = np.radians(alpha).flatten()
        return np.stack((np.cos(alpha_rad), np.sin(alpha_rad)), axis=1)

    @staticmethod
    def get_sample_parameters(
        num: int, spacing: Optional[str] = "cosine"
    ) -> np.ndarray:
        """Returns ``num`` values between 0 and 1 for sampling geometry.

        The parameter can be used to obtain points with linear or
        cosine spacing on a :py:class:`Airfoil`.

        Args:
            num: Number of parameters to sample on the chord-line.
            spacing: Sets the spacing used for the parameters on the
                chord-line between 0-1. Available options are "cosine"
                and "linear". A linear spacing will return an array of
                parameters that are equidistant from each other. Whereas
                a cosine spacing increases accuracy of some aerodynamic
                solvers by increasing the density of parameters close to
                the leading and trailing edges. Defaults to "cosine".
        """

        if spacing == "cosine":
            return 0.5 * (1 - np.cos(np.linspace(0, np.pi, num=num)))
        elif spacing == "linear":
            return np.linspace(0, 1, num=num)
        else:
            raise ValueError(
                f'The supplied `spacing` value of "{spacing}" is '
                'invalid. Please specify either "cosine" or "linear".'
            )
