"""Contains py:class:`BSpline` for splining 2D points."""

from functools import cached_property
from typing import Optional, Union

import numpy as np
import scipy.interpolate as si

from .geom2d import normalize_2d, rotate_2d_90ccw


class BSpline2D:
    """Creates a splined representation of a set of points.

    Args:
        points: A set of 2D row-vectors
        degree: Degree of the spline. Defaults to 3 (cubic spline).
    """

    def __init__(self, points: np.ndarray, degree: Optional[int] = 3):
        self.points = points
        self.degree = degree

    @cached_property
    def spline(self):
        """1D spline representation of :py:attr:`points`.

        Returns:
            Scipy 1D spline representation:
                [0]: Tuple of knots, the B-spline coefficients, degree
                     of the spline.
                [1]: Parametric points, u, used to create the spline
        """
        return si.splprep(self.points.T, s=0.0, k=self.degree)

    def evaluate_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the spline point(s) at ``u``."""
        return np.array(si.splev(u, self.spline[0], der=0), dtype=np.float64).T

    def tangent_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the spline tangent(s) at ``u``."""
        return normalize_2d(
            np.array(si.splev(u, self.spline[0], der=1), dtype=np.float64).T
        )

    def normal_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the spline normals(s) at ``u``."""
        return rotate_2d_90ccw(self.tangent_at(u))
