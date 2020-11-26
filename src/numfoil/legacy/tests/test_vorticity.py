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

# from math import pi, sin

# import numpy as np

# from numfoil.functions import airfoil, v_comp
# from numfoil.panel.vortex_sheet import solve_vorticity as solve_vorticity

# Katz&Plotkin pg267
# def test_vorticity():
#     """..."""
#     Q_inf = v_comp(1, 1)
#     _, _, coor_c = airfoil([0, 0, 0, 0], 5)
#     Gamma, panel_length, _, _ = solve_vorticity(coor_c, Q_inf)
#     ref = np.array([[2.46092],
#                     [1.09374],
#                     [0.70314],
#                     [0.46876],
#                     [0.27344]])
#     calc = Gamma / (pi * np.array([panel_length]).T * sin(1))

#     pass
