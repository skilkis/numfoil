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

"""Contains all :py:class:`Airfoil` class definitions."""

import re
from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import Tuple, Union

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from .geom2d import normalize_2d, rotate_2d_90ccw
from .spline import BSpline2D

# TODO Add NACA5 series Airfoil as a fun nice-to-have feature

AIRFOIL_REPR_REGEX = re.compile(r"[.]([A-Z])\w+")


class Airfoil(metaclass=ABCMeta):
    """Abstract Base Class definition of an :py:class:`Airfoil`.

    The responsibility of an :py:class:`Airfoil` is to provide methods
    that return points on the airfoil upper and lower surface as well as
    the camber-line when given a normalized location along the
    chord-line ``x``. A single point can be obtained by calling these
    methods with a singular float. Alternatively, multiple points can
    be obtained by passing a :py:class:`numpy.ndarray` with n
    row-vectors resulting in a shape: (n, 2).

    The origin of the coordinate system (x=0, y=0, z=0) is located at
    the leading-edge of an airfoil. The positive x-axis is aligned with
    the chord-line of the airfoil, hence the coordinate (1, 0, 0) would
    represent the trailing-edge. The positive z-axis is aligned with
    the positive thickness direction of the airfoil. The positive
    y-axis is then going into the page, hence clock-wise rotations are
    positive. This definition follows that of Katz and Plotkin.
    """

    @property
    @abstractmethod
    def cambered(self) -> bool:
        """Returns if the current :py:class:`Airfoil` is cambered."""

    @abstractmethod
    def camberline_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns camber-line points at the supplied ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)
        """

    @abstractmethod
    def upper_surface_at(
        self, x: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns upper airfoil ordinates at the supplied ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)
        """

    @abstractmethod
    def lower_surface_at(
        self, x: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns lower airfoil ordinates at the supplied ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)
        """

    @staticmethod
    def ensure_1d_vector(x: Union[float, np.ndarray]) -> np.ndarray:
        """Ensures that ``x`` is a 1D vector."""
        x = np.array([x]) if isinstance(x, (float, int)) else x
        if len(x.shape) != 1:
            raise ValueError("Only 1-D np.arrays are supported")
        return x

    def plot(
        self, n_points: int = 100, show: bool = True
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plots the airfoil with ``n_points`` per curve.

        Args:
            n_points: Number of points used per airfoil curve
            show: Determines if the plot window should be launched

        Returns:
            Matplotlib plot objects:

                [0]: Matplotlib Figure instance
                [1]: Matplotlib Axes instance
        """

        # Setting up cosine sampled chord-values
        x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, num=n_points)))

        # Retrieving pts on all curves
        pts_lower = self.lower_surface_at(x)
        pts_upper = self.upper_surface_at(x)

        fig, ax = plt.subplots()
        ax.plot(pts_upper[:, 0], pts_upper[:, 1], label="Upper Surface")
        ax.plot(pts_lower[:, 0], pts_lower[:, 1], label="Lower Surface")
        if self.cambered:
            pts_camber = self.camberline_at(x)
            ax.plot(pts_camber[:, 0], pts_camber[:, 1], label="Camber Line")
        ax.legend(loc="best")
        ax.set_xlabel("Normalized Location Along Chordline (x/c)")
        ax.set_ylabel("Normalized Thickness (t/c)")
        plt.axis("equal")
        plt.show() if show else ()  # Rendering plot window if show is true

        return fig, ax


class NACA4Airfoil(Airfoil):
    """Creates a NACA 4 series :py:class:`Airfoil` from digit input.

    The intented usage is to directly unpack a sequence containing the
    4-digits of the NACA-4 series airfoil definition into the

    Args:
        naca_code: 4-digit NACA airfoil code, i.e. "naca0012" or "0012"

    Keyword Arguments:
        te_closed: Sets if the trailing-edge of the airfoil is closed.
            Defaults to False.

    Attributes:
        max_camber: Maximum camber as a percentage of the chord. Valid
            inputs range from 0-9 % maximum camber. Defaults to 0.
        camber_location: Location of maximum camber in tenths of the
            chord length. A value of 1 would mean 10% of the chord.
            Defaults to 0.
        max_thickness: Maximum thickness as a percentage of the chord.
    """

    def __init__(
        self, naca_code: str, *, te_closed: bool = False,
    ):
        max_camber, camber_location, max_t1, max_t2 = self.parse_naca_code(
            naca_code
        )
        self.max_camber = max_camber / 100
        # The conditional below ensures that the maximum camber is 0.0
        # for symmetric and 0.1 at minimum for cambered airfoils
        if self.max_camber != 0:
            self.camber_location = max(camber_location / 10, 0.1)
        else:
            self.camber_location = 0
        self.max_thickness = float(f".{max_t1}{max_t2}")
        self.te_closed = te_closed

    @property
    def cambered(self) -> bool:
        """Returns if the current :py:class:`Airfoil` is cambered."""
        return self.max_camber != 0 and self.camber_location != 0

    def camberline_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns camber-line points at the supplied ``x``."""
        # Setting up chord-line and camber-line point arrays
        x = self.ensure_1d_vector(x)
        pts_c = np.zeros((x.size, 2))
        pts_c[:, 0] = x

        # Localizing inputs for speed and clarity
        m = self.max_camber
        p = self.camber_location

        if self.cambered:
            fwd, aft = x <= p, x > p  # Indices before and after max ordinate
            pts_c[fwd, 1] = (m / (p ** 2)) * (2 * p * x[fwd] - x[fwd] ** 2)
            pts_c[aft, 1] = (m / (1 - p) ** 2) * (
                (1 - 2 * p) + 2 * p * x[aft] - x[aft] ** 2
            )
        return pts_c

    def camber_tangent_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the camber-line tangent vector at supplied ``x``."""
        # Setting up chord-line and camber-line tangent arrays
        x = self.ensure_1d_vector(x)
        t_c = np.repeat(
            np.array([[1, 0]], dtype=np.float64), repeats=x.size, axis=0
        )

        # Localizing inputs for speed and clarity
        m = self.max_camber
        p = self.camber_location

        if self.cambered:
            fwd, aft = x <= p, x > p  # Indices before and after max ordinate
            t_c[fwd, 1] = (2 * m / p ** 2) * (p - x[fwd])
            t_c[aft, 1] = (2 * m / (1 - p) ** 2) * (p - x[aft])

        return normalize_2d(t_c, inplace=True)

    def camber_normal_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the camber-line normal vector at supplied ``x``.

        Note:
            This method implements a fast 2D Affine Transform.
        """
        return rotate_2d_90ccw(self.camber_tangent_at(x))

    def upper_surface_at(self, x: np.ndarray) -> np.ndarray:
        """Returns upper surface points at the supplied ``x``."""
        c_pts = self.camberline_at(x)
        c_pts += self.offset_vectors_at(x)
        return c_pts

    def lower_surface_at(self, x: np.ndarray) -> np.ndarray:
        """Returns lower surface points at the supplied ``x``."""
        c_pts = self.camberline_at(x)
        c_pts -= self.offset_vectors_at(x)
        return c_pts

    def offset_vectors_at(self, x: np.ndarray) -> np.ndarray:
        """Returns half-thickness magnitude vectors at ``x``."""
        n_c = self.camber_normal_at(x)  # Camber normal-vectors
        y_t = self.half_thickness_at(x)  # Half thicknesses
        return np.multiply(n_c, y_t.reshape(x.size, 1), out=n_c)

    def half_thickness_at(self, x: np.ndarray) -> np.ndarray:
        """Calculates the NACA-4 series 'Half-Thickness' y_t at ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)
        """
        x = self.ensure_1d_vector(x)
        return (self.max_thickness / 0.2) * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * (x ** 2)
            + 0.2843 * (x ** 3)
            - (0.1036 if self.te_closed else 0.1015) * (x ** 4)
        )

    def plot(self, *args, show: bool = True, **kwargs):
        """Specializes the :py:class:`Airfoil` plot with a title."""
        # Turning off plot display to be able to display after the
        # title is added to the plot
        fig, ax = super().plot(*args, **kwargs, show=False)
        ax.set_title(
            "{name} {te_shape} Trailing-Edge Airfoil".format(
                name=self.name, te_shape="Closed" if self.te_closed else "Open"
            )
        )
        plt.show() if show else ()  # Rendering plot window if show is true
        return fig, ax

    @property
    def name(self) -> str:
        """Returns the name of the airfoil from current attributes."""
        return "NACA{m:.0f}{p:.0f}{t:.0f}".format(
            m=self.max_camber * 100,
            p=self.camber_location * 10,
            t=self.max_thickness * 100,
        )

    def __repr__(self) -> str:
        """Overwrites string repr. to include airfoil name."""
        return re.sub(
            AIRFOIL_REPR_REGEX, f".{self.name}Airfoil", super().__repr__()
        )

    @staticmethod
    def parse_naca_code(naca_code: str) -> map:
        """Parses a ``naca_code`` into a map object with 4 entries.

        Note:
            ``naca_code`` can include the prefix "naca" or "NACA".

        Raise:
            ValueError: If a``naca_code`` is supplied with
                missing digits or invalid characters.

        Returns:
            Map object with all 4 digits converted to :py:class:`int`.
        """
        digits = naca_code.upper().strip("NACA")
        if len(digits) == 4 and all(d.isdigit() for d in digits):
            return map(int, digits)
        else:
            raise ValueError("NACA code must contain 4 numbers")


class FileAirfoil(Airfoil):
    """Base class definition of an airfoil read from a file.

    Note:
        Airfoils coordinates are ordered such that the first point is
        the trailing edge, and the points go from the upper surface
        to the lower surface and back to the trailing edge (Counter
        Clockwse).

    Args:
        filepath: Absolute filepath to the airfoil file including
            file extension.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    @abstractmethod
    def parse_file(self) -> np.ndarray:
        """Reads :py:attr:`self.filename` and returns a points array."""

    @cached_property
    def points(self) -> np.ndarray:
        """Returns airfoil ordinate points x, y as row-vectors."""
        return self.parse_file()

    @cached_property
    def le_idx(self) -> np.ndarray:
        """Returns the leading edge index within :py:attr:`points`."""
        return np.argwhere(self.points == np.min(self.points, axis=0))[0, 0]

    @cached_property
    def upper_surface(self) -> BSpline2D:
        """Returns a spline of the upper airfoil surface."""
        return BSpline2D(self.points[self.le_idx :: -1])

    @cached_property
    def lower_surface(self) -> BSpline2D:
        """Returns a spline of the lower airfoil surface."""
        return BSpline2D(self.points[self.le_idx :])

    # TODO consider using geom2d views here
    @cached_property
    def mean_camber_line(self) -> BSpline2D:
        """Returns a spline of the mean-camber line.

        Note:
            The mean camber line is defined as the average distance
            between the top and bottom surfaces when measured normal
            to the chord-line.
        """
        u = np.linspace(0, 1, num=100)

        upper_pts = self.upper_surface.evaluate_at(u)
        lower_pts = self.lower_surface.evaluate_at(u)

        return BSpline2D(0.5 * (upper_pts - lower_pts) + lower_pts)

    @property
    def cambered(self) -> bool:
        """Returns if the current :py:class:`Airfoil` is cambered.

        The algorithm works by flipping the lower surface points and
        checking if the distance between the top and bottom points are
        within a set tolerance.
        """
        u = np.linspace(0, 1, num=10)

        upper_pts = self.upper_surface.evaluate_at(u)
        lower_pts = self.lower_surface.evaluate_at(u)
        lower_pts[:, 1] *= -1  # Flipping the lower surface across the chord

        return not np.allclose(upper_pts, lower_pts, atol=1e-6)

    def camberline_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns camber-line points at the supplied ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)
        """
        return self.mean_camber_line.evaluate_at(u=x)

    def upper_surface_at(
        self, x: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns upper airfoil ordinates at the supplied ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)
        """
        return self.upper_surface.evaluate_at(u=x)

    def lower_surface_at(
        self, x: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns lower airfoil ordinates at the supplied ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)
        """
        return self.lower_surface.evaluate_at(u=x)


class UIUCAirfoil(FileAirfoil):
    """Creates :py:class:`Airfoil` from a UIUC airfoil file."""

    def parse_file(self) -> np.ndarray:
        """UIUC files are ordered correctly and have 1 header line."""
        return np.genfromtxt(self.filepath, skip_header=1)


class ParabolicCamberAirfoil(NACA4Airfoil):
    def __init__(self, eta: float = 0.1):
        self.eta = eta

    def camberline_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns camber-line points at the supplied ``x``."""
        # Setting up chord-line and camber-line point arrays
        x = self.ensure_1d_vector(x)
        pts_c = np.zeros((x.size, 2))
        pts_c[:, 0] = x

        pts_c[..., 1] = 4 * self.eta * (x - x ** 2)
        return pts_c

    @property
    def name(self) -> str:
        """Returns the name of the airfoil from current attributes."""
        return f"ParabolicCamber(e={self.eta:.0f})"
