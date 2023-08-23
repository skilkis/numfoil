"""Task 2: Verification with a thin NACA Airfoil."""

from pathlib import Path
from typing import Dict, Generator, List, Sequence

import numpy as np
import xfoil
from matplotlib import pyplot as plt

from numfoil.geometry import NACA4Airfoil
from numfoil.geometry.airfoil import ParabolicCamberAirfoil
from numfoil.solver.m_linear_vortex import LinearVortex, calc_linear_vortex_im
from numfoil.solver.m_lumped_vortex import LumpedVortex

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

# # Temporary plot of assignment NACA0015 table data
# fig, ax = plt.subplots()
# ax.plot(naca0015_data["x"], naca0015_data["Cp5"])
# ax.invert_yaxis()
# ax.set_ylabel("Pressure Coefficient")
# ax.set_xlabel("Normalized Chordwise Location")
# ax.set_title(r"NACA0015 at $\alpha=5$ [deg]")
# plt.show()


# * ################# plotkin verification funciton ###########################

print('Question 2:')
def delta_cp_plotkin(
    x: np.ndarray, alpha: float, eta: float, c: float = 1.0
) -> np.ndarray:
    """Exact analytical pressure coefficient for a parabolic airfoil."""
    return 4 * np.sqrt((c - x) / x) * np.radians(
        alpha
    ) + 32 * eta / c * np.sqrt((xc := (x / c)) * (1 - xc))



for eta, alpha in [(0.02, 5), (0.06, 7)]:
    camberval = int(eta*100)
    solution = LumpedVortex(
        airfoil=ParabolicCamberAirfoil(eta=eta), n_panels=100, spacing="linear",
        ).solve_for(alpha=alpha)
    fig, ax = solution.plot_delta_cp()

    cps = xfoil.find_pressure_coefficients(f"NACA{camberval}501", alpha=alpha, delete=True)['Cp']
    x = xfoil.find_pressure_coefficients(f"NACA{camberval}501", alpha=alpha, delete=True)['x']
    mid = x.index(min(x))
    up = list(reversed(cps[mid:]))
    low = cps[:mid]
    ax.plot(x[:mid][::5], np.array(up)[::5]-np.array(low)[::5], 'ro', label=f"XFOIL NACA{camberval}501")

    x = np.linspace(0, 1, 25)[1:]
    ax.plot(x, delta_cp_plotkin(x, alpha=alpha, eta=eta), 'xk', label="Exact Solution")

    ax.set_ylim([0, 5])
    ax.legend(loc="best")
    plt.show()
    print(f"Thin Lift Coefficient = {solution.lift_coefficient}")
    cl_xfoil = xfoil.find_coefficients(f"NACA{camberval}501", alpha=alpha, delete=True)['CL']
    print(f"Lift Coefficient NACA{camberval}501 at {alpha} degrees is {cl_xfoil}")
    print(f"relative errror = {(cl_xfoil-solution.lift_coefficient)/cl_xfoil}")
    print()
    # fig.savefig(FIGURE_DIR / f"thin_airfoil_verification_a{alpha}.pdf", bbox_inches="tight")


# for naca_code in ["naca0001", "naca2401"]:
#     for alpha in [0, 3, 5, 8, 10, 13]:
#         solution = LumpedVortex(
#                 airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
#             ).solve_for(alpha=alpha)
#         cl_calc = solution.lift_coefficient
#         cl_xfoil = xfoil.find_coefficients(naca_code, alpha=alpha, delete=True)['CL']
#         print(f"Lift Coefficient {naca_code} at {alpha} degrees is {cl_calc}")
#         print(f" Xfoils CL = {cl_xfoil}")
#         print(f"relative errror = {(cl_xfoil-cl_calc)/cl_xfoil}")
#         print()




# * ###########################################################################


# # * verification of the thick method

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
    # fig.savefig(FIGURE_DIR / f"thick_verif_{naca_code}_alpha{alpha}.pdf", bbox_inches="tight")
    print(f"Lift Coefficient {naca_code} at {alpha} degrees is {solution.lift_coefficient}")

# for naca_code in ["naca0015", "naca2422"]:
#     cl = []
#     for alpha in [3, 7]:
#         solution = LinearVortex(
#                 airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
#             ).solve_for(alpha=alpha)
#         cl.append(solution.lift_coefficient)
#     print(f"lift gradient Cla {naca_code} = {(cl[1] - cl[0])/4}")

# for naca_code in ["naca0015", "naca2422"]:
#     for alpha in [0, 3, 5, 8, 10, 13]:
#         solution = LinearVortex(
#                 airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
#             ).solve_for(alpha=alpha)
#         cl_calc = solution.lift_coefficient
#         cl_xfoil = xfoil.find_coefficients(naca_code, alpha=alpha, delete=True)['CL']
#         print(f"Lift Coefficient {naca_code} at {alpha} degrees is {cl_calc}")
#         print(f" Xfoils CL = {cl_xfoil}")
#         print(f"relative errror = {(cl_xfoil-cl_calc)/cl_xfoil}")
#         print()



# solution = LumpedVortex(
#     airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
# ).solve_for(alpha=alpha)
# fig, ax = solution.plot_lift_gradient()

# xfoil_data = xfoil.find_pressure_coefficients(naca_code, alpha=alpha, delete=True)
# ax.plot(xfoil_data["x"], xfoil_data["Cp"], "xk", markevery=5, label="XFOIL")
# ax.legend()
# ax.legend(loc='best')
# fig.savefig(FIGURE_DIR / f"thick_camber_{naca_code}_alpha{alpha}.pdf", bbox_inches="tight")

