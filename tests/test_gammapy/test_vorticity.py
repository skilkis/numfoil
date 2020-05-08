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

from gammapy.functions import airfoil, v_comp
from gammapy.panel.vortex_sheet import PanelledAirfoil
import numpy as np
from math import sin, pi
from pytest import approx


# Katz&Plotkin pg267
def test_vorticity():
    """..."""
    testfoil = PanelledAirfoil(Naca=[0,0,0,0], n_panels=5, alpha=5, v_inf=1)

    ref = np.array([[2.46092],
                    [1.09374],
                    [0.70314],
                    [0.46876],
                    [0.27344]])

    calc = testfoil.camberline.vorticity.Gamma / (pi * np.array([testfoil.camberline.panel_lengths]).T * sin(testfoil.alpha))

    assert calc == approx(ref, rel=5e-5)
