"""Task 3: Compare Symmetric vs. Asymmetric Airfoil."""

from pathlib import Path
from typing import Dict, Generator, List, Sequence

import numpy as np
import xfoil
from matplotlib import pyplot as plt

from numfoil.geometry import NACA4Airfoil
from numfoil.geometry.airfoil import ParabolicCamberAirfoil
from numfoil.solver.m_linear_vortex import LinearVortex
from numfoil.solver.m_lumped_vortex import LumpedVortex


DATA_DIR = Path(__file__).parent / "reference_data"
FIGURE_DIR = Path(__file__).parent.parent / "docs" / "static"


# * compare symm vs camber

# CP plots

print('Question 3:')
fig, ax = plt.subplots()
alpha = 5
for naca_code in ["naca0015", "naca2415", "naca4415"]:
    solution = LinearVortex(
        airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
    ).solve_for(alpha=alpha)
    fig, ax = solution.plot_pressure_distribution(figax=[fig,ax], label=naca_code)
fig.savefig(FIGURE_DIR / "thick_camber_a5.pdf", bbox_inches="tight")

fig, ax = plt.subplots()
alpha = 5
for naca_code in ["naca0015", "naca2415", "naca4415"]:
    solution = LumpedVortex(
        airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
    ).solve_for(alpha=alpha)
    fig, ax = solution.plot_delta_cp(figax=[fig,ax], label=naca_code)
    ax.set_ylim([0,7])
fig.savefig(FIGURE_DIR / "thin_camber_a5.pdf", bbox_inches="tight")

# lift plots

fig, ax = plt.subplots()
alpha = list(range(13))
tp = True
for naca_code in ["naca0015", "naca2415", "naca4415"]:
    solution = LinearVortex(
        airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
    ).solve_for(alpha=alpha)
    fig, ax = solution.plot_lift_gradient(figax=[fig,ax], label=naca_code, twopi=tp)
    tp = None
fig.savefig(FIGURE_DIR / "thick_camber_cla.pdf", bbox_inches="tight")

fig, ax = plt.subplots()
alpha = list(range(13))
tp = True
for naca_code in ["naca0015", "naca2415", "naca4415"]:
    solution = LumpedVortex(
        airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
    ).solve_for(alpha=alpha)
    fig, ax = solution.plot_lift_gradient(figax=[fig,ax], label=naca_code, twopi=tp)
    tp = None
fig.savefig(FIGURE_DIR / "thin_camber_cla.pdf", bbox_inches="tight")


