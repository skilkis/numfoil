"""Task 5: Show difference between a thick/thin airfoil using XFOIL."""

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

# def splitby(sequence: Sequence, n: int) -> Generator[Sequence, None, None]:
#     """Split a ``sequence`` into chunks of size ``n``."""
#     for i in range(0, len(sequence), n):
#         yield sequence[i : i + n]


# def parse_naca0015_table(
#     filename: str = "naca0015_assignment.txt",
# ) -> Dict[str, List[float]]:
#     """Parses the single column NACA0015 table."""
#     table: Dict[str, List[float]] = {
#         "station": [],
#         "x": [],
#         "Cp0": [],
#         "Cp5": [],
#     }
#     with open(DATA_DIR / filename) as f:
#         lines = f.readlines()
#         # Splitting single column data into chunks to represent columns
#         for chunk in splitby(lines, n=(len(table.keys()) + 1)):
#             station, x, cp0, _, cp5 = chunk
#             for key, value in zip(table.keys(), (station, x, cp0, cp5)):
#                 table[key].append(float(value))
#     return table

# naca0015_data = parse_naca0015_table()

# * ### effect of thickness on Cp ###

# * compare thickness

# CP plots

print('Question 4:')
fig, ax = plt.subplots()
alpha = 5
for naca_code in ["naca0001", "naca0015", "naca0022"]:
    solution = LinearVortex(
        airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
    ).solve_for(alpha=alpha)
    fig, ax = solution.plot_pressure_distribution(figax=[fig,ax], label=naca_code, fixTE=True)
ax.set_ylim([1.1, -2])
fig.savefig(FIGURE_DIR / "thick_tc_a5.pdf", bbox_inches="tight")

fig, ax = plt.subplots()
alpha = 5
for naca_code in ["naca0001", "naca0015", "naca0022"]:
    solution = LumpedVortex(
        airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
    ).solve_for(alpha=alpha)
    fig, ax = solution.plot_delta_cp(figax=[fig,ax], label=naca_code)
    ax.set_ylim([0,7])
fig.savefig(FIGURE_DIR / "thin_tc_a5.pdf", bbox_inches="tight")

# # lift plots

fig, ax = plt.subplots()
alpha = list(range(13))
tp = True
for naca_code in ["naca0001", "naca0015", "naca0022"]:
    solution = LinearVortex(
        airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
    ).solve_for(alpha=alpha)
    fig, ax = solution.plot_lift_gradient(figax=[fig,ax], label=naca_code, twopi=tp)
    tp = None
fig.savefig(FIGURE_DIR / "thick_tc_cla.pdf", bbox_inches="tight")

# fig, ax = plt.subplots()
# alpha = list(range(13))
# tp = True
# for naca_code in ["naca0001", "naca0015", "naca0022"]:
#     solution = LumpedVortex(
#         airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
#     ).solve_for(alpha=alpha)
#     fig, ax = solution.plot_lift_gradient(figax=[fig,ax], label=naca_code, twopi=tp)
#     tp = None
# fig.savefig(FIGURE_DIR / "thin_tc_cla.pdf", bbox_inches="tight")




# print('with camber :')
# fig, ax = plt.subplots()
# alpha = 5
# for naca_code in ["naca2415", "naca2418", "naca2422"]:
#     solution = LinearVortex(
#         airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
#     ).solve_for(alpha=alpha)
#     fig, ax = solution.plot_pressure_distribution(figax=[fig,ax], label=naca_code)
# # fig.savefig(FIGURE_DIR / "thick_tcamber_a5.pdf", bbox_inches="tight")

# fig, ax = plt.subplots()
# alpha = 5
# for naca_code in ["naca2415", "naca2418", "naca2422"]:
#     solution = LumpedVortex(
#         airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
#     ).solve_for(alpha=alpha)
#     fig, ax = solution.plot_delta_cp(figax=[fig,ax], label=naca_code)
#     ax.set_ylim([0,7])

# # lift plots

# fig, ax = plt.subplots()
# alpha = list(range(13))
# tp = True
# for naca_code in ["naca2415", "naca2418", "naca2422"]:
#     solution = LinearVortex(
#         airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
#     ).solve_for(alpha=alpha)
#     fig, ax = solution.plot_lift_gradient(figax=[fig,ax], label=naca_code, twopi=tp)
#     tp = None
# # fig.savefig(FIGURE_DIR / "thick_tcamb_cla.pdf", bbox_inches="tight")

# fig, ax = plt.subplots()
# alpha = list(range(13))
# tp = True
# for naca_code in ["naca2415", "naca2418", "naca2422"]:
#     solution = LumpedVortex(
#         airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
#     ).solve_for(alpha=alpha)
#     fig, ax = solution.plot_lift_gradient(figax=[fig,ax], label=naca_code, twopi=tp)
#     tp = None
