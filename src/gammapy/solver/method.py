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

"""Contains the ABC definition of a panel method solver."""

from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Union

import numpy as np

from gammapy.geometry import Airfoil
from gammapy.geometry.panel import Panel2D

from .solution import FlowSolution


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

    # Responsible for transforming the output of a panel method
    # solution (circulation strengths) into physical quantities and
    # flow coefficients.
    __solution_class__ = FlowSolution

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

    # TODO reduce the only input to alpha, all others are redundant
    # TODO a solution should have cp, cl, delta cp THATS IT
    def solve_for(
        self,
        density: float,
        velocity: float,
        alpha: Union[float, Sequence[float]],
        plot: bool = False,
    ) -> FlowSolution:
        """."""
        # TODO add self to solution_class so it can access properties
        return self.__solution_class__(
            panels=self.panels,
            circulations=self.get_circulation_strengths(velocity, alpha),
            density=density,
            velocity=velocity,
            alpha=alpha,
        )

    def get_circulation_strengths(
        self, velocity: float, alpha: Union[float, Sequence[float]]
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
            velocity: Free-stream flow velocity
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
        v_vector = self.get_velocity_vector(velocity, alpha)

        # Obtaining the Right-Hand-Side RHS by taking the dot product
        # with the velocity vector. Resultant dimensions of the RHS
        # vector is (N_panels, 2), (2, N_alpha) -> (N_panels, N_alpha)
        rhs = self.unit_rhs_vector @ -v_vector.T

        # Solution of the system has dimensions (N_panels, N_alpha)
        return np.linalg.solve(a_matrix, rhs)

    # TODO change to get_flow_direction
    @staticmethod
    def get_velocity_vector(
        velocity: float, alpha: Union[float, Sequence[float]]
    ) -> np.ndarray:
        """Returns velocity components U and W on the x and z axis.

        If an iterable sequence of N Angle of Attacks (AoAs) are given
        then the return statement will be a set of 2D row-vectors with
        shape (N, 2).

        Args:
            velocity: Freestrean velocity in SI meter per second
            alpha: Airfoil Angle of Attack (AoA) in SI degree
        """
        # Converting alpha to radians and ensuring that it is always 1D
        alpha_rad = np.radians(alpha).flatten()
        return velocity * np.stack(
            (np.cos(alpha_rad), np.sin(alpha_rad)), axis=1
        )

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
