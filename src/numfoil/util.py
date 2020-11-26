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

"""Contains utility functions for pressure coefficient analysis."""

from typing import Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d


def delta_cp_from_cp(
    x_values: Sequence[float], pressure_coefficients: Sequence[float], num=1000
):
    """Assuming CCW airfoil TE -> Upper -> LE -> Lower -> TE."""
    (x_top, cp_top), (x_bot, cp_bot) = split_at_le(
        x_values, pressure_coefficients
    )

    interp1d_kwargs = {"kind": "cubic", "fill_value": "extrapolate"}

    f_top = interp1d(
        tuple(reversed(x_top)), tuple(reversed(cp_top)), **interp1d_kwargs
    )
    f_bot = interp1d(x_bot, cp_bot, **interp1d_kwargs)

    sample_x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, num=num)))

    return sample_x, f_bot(sample_x) - f_top(sample_x)


def split_at_le(
    x_values: Sequence[float], data: Sequence[float]
) -> Tuple[Tuple[Sequence[float], Sequence[float]]]:
    """Splits ``data`` at the leading-edge of the airfoil.

    The leading-edge is assumed to be the first occurance of the
    minimum in ``x_values``.
    """
    le_idx = x_values.index(min(x_values))
    return (
        (x_values[: le_idx + 1], data[: le_idx + 1]),
        (x_values[le_idx + 1 :], data[le_idx + 1 :]),
    )
