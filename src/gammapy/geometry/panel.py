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

from .vector2d import (
    is_row_vector,
    magnitude_2d,
    normalize_2d,
    rotate_2d_90ccw,
)


class Panel2D(np.ndarray):
    """Creates n-1 panels from a set of n points.

    Note:
        A :py:class:`Panel2D` instantiation behaves exactly the same as
        a :py:class:`numpy.ndarray` instance. Therefore, if a set of n
        row-vectors is the input, then each property will return an
        array of n-1 rows.

    Args:
        array: A 2D Numpy array containing n row-vectors.
    """

    def __new__(cls, array: Union[Sequence[Tuple[float, float]], np.ndarray]):
        """Creates a :py:class:`Panel2D` instance from ``array``.

        Args:
            array: A 2D Numpy array containing n row-vectors.
        """
        array = np.array(array) if not isinstance(array, np.ndarray) else array
        assert is_row_vector(array)
        if array.shape[0] < 2:
            raise ValueError("A panel requires at least 2 points")
        return np.asarray(array, dtype=np.float64).view(cls)

    @property
    def nodes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a view on the start and end nodes of all panels."""
        return self[:-1].view(np.ndarray), self[1:].view(np.ndarray)

    @property
    def tangents(self) -> np.ndarray:
        """Returns unit tangent vectors of all panels."""
        starts, ends = self.nodes
        return normalize_2d(ends - starts, inplace=True)

    @property
    def normals(self) -> np.ndarray:
        """Returns unit normal vectors of all panels."""
        return rotate_2d_90ccw(self.tangents)

    @property
    def lengths(self) -> np.ndarray:
        """Returns length of all panels."""
        starts, ends = self.nodes
        return magnitude_2d(ends - starts)

    def points_at(self, u: float) -> np.ndarray:
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

        pts = self  # Panel points is simply the input array
        n_pts = self.shape[0]
        pts_mid = self.points_at(0.5)

        fig, ax = plt.subplots()
        ax.scatter(pts[:, 0], pts[:, 1], marker="o", label="Nodes")
        ax.plot(pts[:, 0], pts[:, 1], label="Edges", zorder=1)
        for vector in (self.tangents, self.normals):
            ax.quiver(
                pts_mid[:, 0],
                pts_mid[:, 1],
                vector[:, 0],
                vector[:, 1],
                angles="xy",
                scale=None,
                width=2e-3,
                zorder=2,
            )
        plt.scatter(pts_mid[:, 0], pts_mid[:, 1], s=2, color="black", zorder=2)
        ax.legend(loc="best")
        ax.set_xlabel("Principal Axis [-]")
        ax.set_ylabel("Secondary Axis [-]")
        ax.set_title(f"Panel Geometry {n_pts} Nodes, {n_pts-1} Edges")
        plt.axis("equal")

        plt.show() if show else ()  # Rendering plot window if show is true

        return fig, ax
