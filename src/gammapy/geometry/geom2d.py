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

"""Contains primitive geometric definitions for points and vectors."""

from typing import Sequence, Tuple, Union

import numpy as np

from .vector2d import is_row_vector, magnitude_2d, normalize_2d


class Geom2D(np.ndarray):
    """Defines the primitive row-vector as a geometric array."""

    def __new__(cls, array: Union[Sequence[Tuple[float, float]], np.ndarray]):
        """Creates a :py:class:`Geom2D` instance from ``array``.

        Args:
            array: A 2D Numpy array containing n row-vectors
        """
        array = np.array(array) if not isinstance(array, np.ndarray) else array
        assert is_row_vector(array)
        return np.asarray(array, dtype=np.float64).view(cls)


class Point2D(Geom2D):
    """Defines a point in 2D space."""

    @property
    def x(self) -> np.ndarray:
        """Returns the x coordinate(s) of :py:class:`Point2D`."""
        return self[..., 0].view(np.ndarray)

    @property
    def y(self) -> np.ndarray:
        """Returns the y coordinate(s) of :py:class:`Point2D`."""
        return self[..., 1].view(np.ndarray)

    def __sub__(self, other) -> object:
        """Overloads subtract magic method to allow vector creation.

        When two :py:class:`Point2D` objects are subtracted from
        one another a vector is created as follows::

            a = Point2D([0, 0])
            b = Point2D([1, 1])

            b - a
            Vector2D([1, 1])  # A Vector2D object is created from a to b

            a - b
            Vector2D([1, 1])  # A Vector2D object is created from b to a

        However, if a simple scalar is added to the :py:class:`Point2D`
        object it will return a `Point2D` object as expected::

            a = Point2D([0, 0])
            a + 2

            Point2D([2, 2])
        """
        if isinstance(other, Point2D):
            return super().__sub__(other).view(Vector2D)
        else:
            return super().__sub__(other)


class Vector2D(Point2D):
    """Defines a vector in 2D space."""

    @property
    def magnitude(self) -> np.ndarray:
        """Calculates the length (magnitude) of py:class:`Vector2D`."""
        return magnitude_2d(self).view(np.ndarray)

    @property
    def normalized(self) -> np.ndarray:
        """Returns the nit-vector(s) of :py:class:`Vector2D`."""
        return normalize_2d(self)

    # Subtracting entities with
    __sub__ = Geom2D.__sub__
