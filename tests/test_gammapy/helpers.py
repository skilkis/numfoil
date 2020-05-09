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

"""Collections of test helpers for the GammaPy package."""

import os
from functools import partial
from typing import Generator

import numpy as np
import scipy.interpolate as si


def get_naca_airfoils(directory: str) -> Generator[str, None, None]:
    """Retrieves the NACA 4 series airfoil files in ``directory``."""
    files = filter(
        lambda f: os.path.isfile(os.path.join(directory, f)),
        os.listdir(directory),
    )
    for filename in files:
        name, ext = os.path.splitext(filename)
        try:
            n_digits = len(tuple(map(int, name.upper().strip("NACA"))))
            if ext.upper() == ".DAT" and n_digits == 4:
                yield filename
        except ValueError:
            pass


def calc_curve_error(
    crv_1: np.ndarray, crv_2: np.ndarray, num: int = 100
) -> float:
    """Calculates the error between ``crv_1`` and ``crv_2``.

    A useful function to compare two curves which are represented by a
    different number of points or misaligned set of points. Internally
    a spline is created from both set of points and the two curves
    are then subsequently sampled with the same parameter vector, u.

    Args:
        crv_1: First curve represented by row-vectors (n, 2)
        crv_1: Second curve represented by row-vectors (n, 2)
        num: Number of sampling points on the spline

    Returns:
        Residual sum of squares (RSS) error of ``crv_1`` and ``crv_2``.
    """
    # Creating splines from both set of points
    splprep = partial(si.splprep, s=0.0, k=3)
    spl_1 = splprep((crv_1[:, 0], crv_1[:, 1]))[0]
    spl_2 = splprep((crv_2[:, 0], crv_2[:, 1]))[0]

    # Creating new points with parameter sampling
    splev = partial(si.splev, der=0)  # Returns the points at u
    u = np.linspace(0, 1, num=num)
    pts_1 = np.array(splev(u, spl_1))
    pts_2 = np.array(splev(u, spl_2))

    return np.sum((pts_1 - pts_2) ** 2)
