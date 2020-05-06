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
from functools import lru_cache
from typing import Optional, Tuple, Union

import numpy as np

# from .transforms import AffineTransform2D

# TODO Add NACA5 series Airfoil as a fun nice-to-have feature
# TODO Add the ability to import an arbitrary airfoil from a data-file
# TODO use bezier package on PyPi to have camber-line calculations

AIRFOIL_REPR_REGEX = r"[.]([A-Z])\w+"


class Airfoil(metaclass=ABCMeta):
    """Abstract Base Class definition of an Airfoil.

    The origin of the coordinate system (x=0, y=0, z=0) is located at
    the leading-edge of an airfoil. The positive x-axis is aligned with
    the chord-line of the airfoil, hence the coordinate (1, 0, 0) would
    represent the trailing-edge.

    Note:
        Although only 2 axes are necessary to represent an airfoil, the
        coordinate system follows that of Katz and Plotkin.

    """

    @abstractmethod
    def camber_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the camber-line z-value at the supplied ``x``."""

    @abstractmethod
    def surfaces_at(
        self, x: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns airfoil surface ordinates at the supplied ``x``."""

    @staticmethod
    @lru_cache
    def get_sample_points(
        n_points: int, spacing: Optional[str] = "cosine"
    ) -> np.ndarray:
        """Returns a number of ``n_points`` on the chord-line.

        These points will be for

        Args:
            n_points: Number of points to sample on the chord-line.
            spacing: Sets the spacing used for the points on the
                chord-line. Available options are "cosine" and "linear".
                A linear spacing will return an array of points that are
                equidistant from each other. Whereas a cosine spacing
                will increase the accuracy of some aerodynamic solvers
                by increasing the density of points close to the
                leading and trailing edges. Defaults to "cosine".
        """
        if spacing == "cosine":
            sample = np.linspace(0, np.pi, num=n_points)
            return 0.5 * (1 - np.cos(sample))
        elif spacing == "linear":
            return np.linspace(0, 1, num=n_points)
        else:
            raise ValueError(
                f'The supplied `spacing` value of "{spacing}" is invalid. '
                'Please specify either "cosine" or "linear".'
            )


class NACA4Airfoil(Airfoil):
    """Creates a NACA 4 series :py:class:`Airfoil` from digit input.

    The intented usage is to directly unpack a sequence containing the
    4-digits of the NACA-4 series airfoil definition into the


    Args:
        max_camber: Maximum camber as a percentage of the chord. Valid
            inputs range from 0-9 % maximum camber. Defaults to 0.
        camber_location: Location of maximum camber in tenths of the
            chord length. A value of 1 would mean 10% of the chord.
            Defaults to 0.
        max_t1: First digit of the location of maximum thickness as a
            percentage of the airfoil chord. Defaults to 1.
        max_t2: Second digit of the location of maximum thickness as a
            percentage of the airfoil chord. Defaults to 2.

    Keyword Arguments:
        te_closed: Sets if the trailing-edge of the airfoil is closed.
            Defaults to False.
    """

    def __init__(
        self,
        max_camber: int = 0,
        camber_location: int = 0,
        max_t1: int = 1,
        max_t2: int = 2,
        *,
        te_closed: bool = False,
    ):
        self.max_camber = max_camber / 100
        if self.max_camber != 0:
            self.camber_location = max(camber_location / 10, 0.1)
        else:
            self.camber_location = 0
        self.max_thickness = float(f".{max_t1}{max_t2}")
        self.te_closed = te_closed

    def camber_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the camber-line z-value at the supplied ``x``."""
        # Setting up chord-line and camber-line point arrays
        x = self.ensure_vector(x)
        z_c = np.zeros(x.shape)

        # Localizing inputs for speed and clarity
        m = self.max_camber
        p = self.camber_location

        if self.cambered:
            fwd, aft = x <= p, x > p  # Indices before and after max ordinate
            z_c[fwd] = (m / (p ** 2)) * (2 * p * x[fwd] - x[fwd] ** 2)
            z_c[aft] = (m / (1 - p) ** 2) * (
                (1 - 2 * p) + 2 * p * x[aft] - x[aft] ** 2
            )
        return z_c

    # TODO Test non-cambered airfoil
    # TODO Test cambered airfoil
    # TODO Test single x and vector x
    def camber_tangent_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the camber-line tangent vector at supplied ``x``."""
        # Setting up chord-line and camber-line tangent arrays
        x = self.ensure_vector(x)
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

        # Obtaining the magnitude of each vector
        magnitude = np.linalg.norm(t_c, axis=1).reshape((x.size, 1))

        # Transforming the camber-line vectors to unit vectors in-place
        np.divide(t_c, magnitude, out=t_c)
        return t_c

    def camber_normal_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the camber-line normal vector at supplied ``x``.

        This method implements a 2D Affine Transform
        """
        return np.rot90(self.camber_tangent_at(x), 1)
        # n_c = np.flip(self.camber_tangent_at(x), axis=1)
        # return np.multiply(n_c, np.array([[-1, 1]], dtype=np.float64), out=n_c)

    def surfaces_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = self.ensure_vector(x)  # Sample x-values
        c_pts = np.stack((x, self.camber_at(x)), axis=1)  # Camber-points
        n_c = self.camber_normal_at(x)  # Camber normal-vectors
        y_t = self.half_thickness_at(x, self.max_thickness)  # Half thicknesses

        offset_vectors = np.multiply(n_c, y_t.reshape((x.size, 1)), out=n_c)
        upper_pts = c_pts + offset_vectors
        lower_pts = c_pts - offset_vectors
        return upper_pts, lower_pts

    # TODO move into mixins if possible else leave it in Airfoils
    @staticmethod
    def ensure_vector(x):
        x = np.array([x]) if isinstance(x, (float, int)) else x
        assert len(x.shape) == 1, "Only 1-D np.arrays are supported"
        return x

    @property
    def cambered(self) -> bool:
        return self.max_camber != 0 and self.camber_location != 0

    def half_thickness_at(self, x: np.ndarray, t: float) -> np.ndarray:
        """Calculates the NACA-4 series 'Half-Thickness' y_t at ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)
            t: Maximum thickness as a fraction of the chord
        """
        x = self.ensure_vector(x)
        return (t / 0.2) * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * (x ** 2)
            + 0.2843 * (x ** 3)
            - (0.1036 if self.te_closed else 0.1015) * (x ** 4)
        )

    def __repr__(self):
        obj_repr = ".NACA{m:.0f}{p:.0f}{t:.0f}Airfoil".format(
            m=self.max_camber * 100,
            p=self.camber_location * 10,
            t=self.max_thickness * 100,
        )
        return re.sub(AIRFOIL_REPR_REGEX, obj_repr, super().__repr__())


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    a = NACA4Airfoil(2, 5, 1, 2, te_closed=True)
    x = a.get_sample_points(2000, "cosine")
    n_c = a.camber_normal_at(x)
    # s = a.surfaces_at(x)
    # plt.style.use("ggplot")
    # plt.plot(s[0][:, 0], s[0][:, 1], s[1][:, 0], s[1][:, 1])
    # c_pts = np.stack((x, a.camber_at(x)), axis=1)  # Camber-points
    # plt.plot(c_pts[:, 0], c_pts[:, 1])
    # # plt.axis("equal")
    # plt.show()
