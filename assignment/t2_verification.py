"""Task 2: Verification with a thin NACA Airfoil."""

from math import pi
from pathlib import Path

from numpy.core.function_base import linspace
from src.numfoil.solver.m_lumped_vortex import LumpedVortex
from typing import Dict, Generator, List, Sequence

import numpy as np
import xfoil
from matplotlib import pyplot as plt

from numfoil.geometry import NACA4Airfoil
from numfoil.legacy.panel.thick import Solver, ThickPanelledAirfoil
from numfoil.solver.m_linear_vortex import LinearVortex




# * ################# Stuff to get the 0015 reference data ####################
DATA_DIR = Path(__file__).parent / "reference_data"
FIGURE_DIR = Path(__file__).parent.parent / "docs" / "static"

def splitby(sequence: Sequence, n: int) -> Generator[Sequence, None, None]:
    """Split a ``sequence`` into chunks of size ``n``."""
    for i in range(0, len(sequence), n):
        yield sequence[i : i + n]


def parse_naca0015_table(
    filename: str = "naca0015_assignment.txt",
) -> Dict[str, List[float]]:
    """Parses the single column NACA0015 table."""
    table: Dict[str, List[float]] = {
        "station": [],
        "x": [],
        "Cp0": [],
        "Cp5": [],
    }
    with open(DATA_DIR / filename) as f:
        lines = f.readlines()
        # Splitting single column data into chunks to represent columns
        for chunk in splitby(lines, n=(len(table.keys()) + 1)):
            station, x, cp0, _, cp5 = chunk
            for key, value in zip(table.keys(), (station, x, cp0, cp5)):
                table[key].append(float(value))
    return table

naca0015_data = parse_naca0015_table()

# Temporary plot of assignment NACA0015 table data
fig, ax = plt.subplots()
ax.plot(naca0015_data["x"], naca0015_data["Cp5"])
ax.invert_yaxis()
ax.set_ylabel("Pressure Coefficient")
ax.set_xlabel("Normalized Chordwise Location")
ax.set_title(r"NACA0015 at $\alpha=5$ [deg]")
plt.show()


# * ################# plotkin verification funciton ###########################

def delta_cp_plotkin(x: np.ndarray, eta: float, c: float = 1.0) -> np.ndarray:
    """Exact analytical pressure coefficient for a parabolic airfoil."""
    return 4 * np.sqrt((c - x) / x) + 32 * eta / c * np.sqrt(
        (xc := (x / c)) * (1 - xc)
    )

# * #########################idk what this is##################################

def get_xfoil_cl(airfoil, alphas):
    return [
        xfoil.find_coefficients("naca0012", alpha=alpha, delete=True)["CL"]
        for alpha in alphas
    ]
# * ###########################################################################





# * verification of the thick method
print('Question 2:')
for naca_code, alpha in [("naca0015", 5), ("naca2422", 10)]:
    solution = LinearVortex(
        airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
    ).solve_for(alpha=alpha)
    fig, ax = solution.plot_pressure_distribution()

    xfoil_data = xfoil.find_pressure_coefficients(naca_code, alpha=alpha, delete=True)
    ax.plot(xfoil_data["x"], xfoil_data["Cp"], "xk", markevery=5, label="XFOIL")
    ax.legend()
    ax.legend(loc='best')
    fig.savefig(FIGURE_DIR / f"thick_verif_{naca_code}_alpha{alpha}.pdf", bbox_inches="tight")



# * compare symm vs camber

print('Question 3:')

naca_code = '0015'
alpha = 5
solution = LumpedVortex(
    airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
).solve_for(alpha=alpha)
fig, ax = solution.plot_lift_gradient()

xfoil_data = xfoil.find_pressure_coefficients(naca_code, alpha=alpha, delete=True)
ax.plot(xfoil_data["x"], xfoil_data["Cp"], "xk", markevery=5, label="XFOIL")
ax.legend()
ax.legend(loc='best')
fig.savefig(FIGURE_DIR / f"thick_camber_{naca_code}_alpha{alpha}.pdf", bbox_inches="tight")
