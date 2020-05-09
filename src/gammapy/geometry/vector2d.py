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


"""Contains functions for performing operations on 2D vectors.

A 2D row-vector is defined as one with 1 row and 2 columns. This
function can also handle a set of n row-vectors given as a 2D
array with n rows and 2 columns. The transform will be applied to
each vector individually using the inner matrix (dot) product.
"""


import numpy as np

__all__ = ["rotate_2d_90ccw", "magnitude_2d", "normalize_2d"]


R_90CCW_MATRIX = np.array([[0, 1], [-1, 0]])


def rotate_2d_90ccw(array: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Rotates 2D row-vector(s) 90 degrees counter-clockwise.

    Args:
        array: An array of row-vector(s) with shape (n, 2)
        inplace: If the normalization should be performed inplace
            on the ``array`` object.

    Returns:
        Transformed row-vector(s) with dimension (n, 2).
    """
    assert is_row_vector(array)
    return np.dot(a=array, b=R_90CCW_MATRIX, out=array if inplace else None)


def magnitude_2d(array: np.ndarray) -> np.ndarray:
    """Calculates the magnitude of 2D row-vector(s).

    Args:
        array: An array of row-vector(s) with shape (n, 2)

    Returns:
        A column-vector of size (n, 1) containing n magnitudes.
    """
    assert is_row_vector(array)
    if len(array.shape) < 2:  # array is a 1D vector
        return np.linalg.norm(array).reshape(1, 1)
    else:
        return np.linalg.norm(array, axis=1).reshape((array.shape[0], 1))


def normalize_2d(array: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Normalizes 2D row-vector(s) into unit-vector(s).

    Args:
        array: An array of row-vector(s) with shape (n, 2)
        inplace: If the normalization should be performed inplace
            on the ``array`` object.

    Returns:
        Normalized row-vector(s) with dimension (n, 2).
    """
    assert is_row_vector(array)
    return np.divide(
        array, magnitude_2d(array), out=array if inplace else None,
    )


def is_row_vector(array: np.ndarray) -> bool:
    """Returns ``True`` if ``array`` is contains 2D row-vector(s).

    Raises:
        ValueError: If an ``array`` is 3D or contains column-vector(s)
    """
    if len(array.shape) == 2 and array.shape[1] != 2:
        raise ValueError(
            "The input `array` must contain 2D row-vectors with shape (n, 2)"
        )
    return True
