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

"""Contains class definitions for creating panels from points."""

from typing import Sequence, Tuple, Union

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from .geom2d import Geom2D, Point2D, Vector2D
from .vector2d import rotate_2d_90ccw


class Panel2D(Geom2D):
    """Creates n panels from a set of n+1 :py:class:`Point2D` arrays.

    The :py:class:`Panel2D` can either be indexed using Numpy syntax
    in which case the underlying py:class:`Point2D` arrays will be
    returned::

        >>> panels = Panel2D([(0, 0), (1, 0), (2, 0)])
        >>> panels[:, :]
        Point2D([[0, 0], [1, 0], [2, 0]])

    Alternatively, if an integer index is used then the ith panel
    will be returned::

        >>> panels = Panel2D([(0, 0), (1, 0), (2, 0)])
        >>> panels[1]
        Panel2D([[1, 0], (2, 0)])

    Note:
        A :py:class:`Panel2D` instantiation behaves exactly the same as
        a :py:class:`numpy.ndarray` instance. Therefore, if a set of n
        row-vectors is the input, then each property will return an
        array of n-1 rows.

    Args:
        array: A 2D Numpy array containing n row-vectors.
    """

    def __new__(
        cls, array: Union[Sequence[Tuple[float, float]], np.ndarray, Point2D]
    ):
        """Creates a :py:class:`Panel2D` instance from ``array``.

        Args:
            array: A 2D Numpy array containing n row-vectors that
                describes a set of 2D points through which the
                panels should be built.
        """
        array = super().__new__(cls, array)
        array = np.array(array) if not isinstance(array, np.ndarray) else array
        if array.shape[0] < 2:
            raise ValueError("A panel requires at least 2 points")
        return array

    @property
    def n_panels(self) -> int:
        """Returns the number of panels of the current instance."""
        rows, cols = self.shape
        return rows - 1

    @property
    def nodes(self) -> Tuple[Point2D, Point2D]:
        """Returns the start and end nodes of all panels.

        Note:
            Due to :py:meth:`__getitem__` being overriden, the
            underlying :py:class:`Point2D` nodes can be accessed
            directly when using a slice.
        """
        return self[:-1], self[1:]

    @property
    def tangents(self) -> Vector2D:
        """Returns unit tangent vectors of all panels."""
        starts, ends = self.nodes
        return (ends - starts).normalized

    @property
    def normals(self) -> Vector2D:
        """Returns unit normal vectors of all panels."""
        return rotate_2d_90ccw(self.tangents)

    @property
    def angles(self) -> np.ndarray:
        """Returns the panel angle w.r.t the primary-axis in radians."""
        tangents = self.tangents
        angles = np.arctan2(tangents.y, tangents.x)[..., None]
        return angles.view(np.ndarray)

    @property
    def lengths(self) -> np.ndarray:
        """Returns length of all panels."""
        starts, ends = self.nodes
        return (ends - starts).magnitude

    def points_at(self, u: float) -> Point2D:
        """Places points at normalized length ``u`` along each panel.

        Args:
            u: Normalized length along a panel edge. 0 represents the
                start node and 1 represents the end node of a panel.
        """
        tangents = self.tangents
        offset_lengths = u * self.lengths
        offset_vectors = np.multiply(offset_lengths, tangents, out=tangents)
        starts, _ = self.nodes
        return starts + offset_vectors

    def plot(
        self, show: bool = True
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plots the current :py:class:`Panel2D` geometry.

        Args:
            show: Determines if the plot window should be launched

        Returns:
            Matplotlib plot objects:

                [0]: Matplotlib Figure instance
                [1]: Matplotlib Axes instance
        """

        # Panel points is simply the input viewed as a Point2D object
        pts = self.view(Point2D)
        n_pts = self.shape[0]
        pts_mid = self.points_at(0.5)

        fig, ax = plt.subplots()
        ax.scatter(pts.x, pts.y, marker="o", label="Nodes")
        ax.plot(pts.x, pts.y, label="Edges", zorder=1)
        for vector in (self.tangents, self.normals):
            ax.quiver(
                pts_mid.x,
                pts_mid.y,
                vector.x,
                vector.y,
                angles="xy",
                scale=None,
                width=2e-3,
                zorder=2,
            )
        ax.scatter(pts_mid.x, pts_mid.y, s=2, color="black", zorder=2)
        ax.legend(loc="best")
        ax.set_xlabel("Principal Axis [-]")
        ax.set_ylabel("Secondary Axis [-]")
        ax.set_title(f"Panel Geometry {n_pts} Nodes, {n_pts-1} Edges")
        plt.axis("equal")

        plt.show() if show else ()  # Rendering plot window if show is true

        return fig, ax

    def __getitem__(self, item) -> np.ndarray:
        """Returns either the n-th panel or :py:class:`Point2D` objects.

        If ``item`` is an integer then the n-th :py:class:`Panel2D`
        (edge) object will be returned. Otherwise, the underlying
        :py:class:`Point2D` nodes are returned.

        This also supports iterating over panels as follows::

            for panel in Panel2D([(0, 0), (1, 0), (2, 0)]):
                pass  # 2 iterations will commence
        """
        if isinstance(item, int):
            n = item
            if -self.n_panels <= n < self.n_panels:
                start = n if n >= 0 else (self.n_panels) + n
                end = start + 2
                return super().__getitem__(slice(start, end, 1))
            else:
                raise IndexError
        else:
            return super().__getitem__(item).view(Point2D)
