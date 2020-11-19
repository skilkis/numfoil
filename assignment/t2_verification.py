"""Task 2: Verification with a thin NACA Airfoil."""

import csv
from math import pi
from pathlib import Path
from typing import Dict, Generator, List, Sequence

import numpy as np
import xfoil
from matplotlib import pyplot as plt

from gammapy.geometry.airfoil import NACA4Airfoil
from gammapy.legacy.panel.thick import Solver, ThickPanelledAirfoil
from gammapy.solver.m_linear_vortex import LinearVortex

# from xfoil import find_pressure_coefficients

# alpha = 1
# values1 = find_pressure_coefficients("naca0016", alpha, delete=True)
# values2 = find_pressure_coefficients("naca0012", alpha, delete=True)

# plt.style.use("ggplot")
# plot = plt.plot(values1["x"], values1["Cp"], label="NACA-0016")
# plt.plot(values2["x"], values2["Cp"], label="NACA-0012")
# plt.xlabel("Chord Fraction (x/c)")
# plt.ylabel("Pressure Coefficient [-]")
# plt.title(r"NACA Airfoils at $\alpha={alpha}$".format(alpha=alpha))
# plt.legend(loc="best")
# plt.gca().invert_yaxis()
# plt.show()


DATA_DIR = Path(__file__).parent / "reference_data"


def splitby(sequence: Sequence, n: int) -> Generator[Sequence, None, None]:
    """Split a ``sequence`` into chunks of size ``n``."""
    for i in range(0, len(sequence), n):
        yield sequence[i : i + n]


def parse_naca0015_table(
    filename: str = "naca0015_assignment.txt",
) -> Dict[str, List[float]]:
    """Parses the single column NACA0015 table."""
    table = {
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


def get_xfoil_cp(nacafoil):
    with open(DATA_DIR / f"{nacafoil}.txt", "r") as f:
        reader = csv.reader(f, skipinitialspace=True, delimiter=" ")
        Cp = np.array([[0, 0]])
        for i in range(3):
            next(reader)
        for row in reader:
            Cp = np.append(Cp, [[float(row[0]), float(row[2])]], axis=0)
    return np.delete(Cp, 0, 0)


def get_data(Naca="0012", alpha=0):
    foil = ThickPanelledAirfoil(Naca=Naca, n_panels=100)
    # foil.panels.plt()
    solver = Solver(foil.panels)
    Cp = solver.solve_Cp(alpha=alpha, plot=False)
    Cl = solver.get_cl(alpha=alpha)
    return Cp, Cl, foil


# * ### create xfoil data ###
# import xfoil
# alfas = [0,3,5,8,10,13,15]
# xfoil.call(NACA=True, airfoil='naca0012', alfas=alfas, output='Polar')
# xfoil.call(NACA=True, airfoil='naca4412', alfas=alfas, output='Polar')

xfoil_0012 = xfoil.find_pressure_coefficients("naca0012", alpha=3, delete=True)

# * ### verification Cp ###
Cp1, cl1, foil1 = get_data(Naca="0012", alpha=3)
fig, ax = plt.subplots()
ax.plot(
    [i[0] for i in foil1.panels.collocation_points], Cp1, "k", label="GammaPy"
)
ax.plot(xfoil_0012["x"], xfoil_0012["Cp"], "xr", markevery=5, label="XFOIL")
ax.invert_yaxis()
ax.set_ylabel(r"$C_p$")
ax.set_xlabel("x/c")
plt.legend(loc="best")
plt.show()
plt.style.use("ggplot")

# * ### verification Cl ###


def get_xfoil_cl(airfoil, alphas):
    return [
        xfoil.find_coefficients("naca0012", alpha=alpha, delete=True)["CL"]
        for alpha in alphas
    ]


alfas = [0, 3, 5, 8, 10, 13, 15]


def get_cls(naca):
    cl = []
    for a in alfas:
        _, Cl, _ = get_data(Naca=naca, alpha=a)
        cl.append(Cl)
    return np.reshape(cl, (len(alfas), 1))


cl0012 = get_cls("0012")
cl4412 = get_cls("4412")

cl0012_xfoil = get_xfoil_cl("naca0012", alfas)
cl4412_xfoil = get_xfoil_cl("naca4412", alfas)


fig, ax = plt.subplots()
ax.plot(alfas, cl0012_xfoil, "k", label=" Xfoil")
ax.plot(alfas, cl0012, "b", label=" GammaPy")
# ax.plot(*zip(*cl4412_xfoil), "g", label="NACA4412 Xfoil")
# ax.plot([i[0] for i in cl4412_xfoil], cl4412, "r", label="NACA4412 GammaPy")
ax.set_ylabel(r"$C_l$")
ax.set_xlabel(r"$\alpha$")
plt.legend(loc="best")
plt.show()
plt.style.use("ggplot")

fig, ax = plt.subplots()
ax.plot(alfas, cl4412_xfoil, "g", label=" Xfoil")
ax.plot(alfas, cl4412, "r", label=" GammaPy")
ax.set_ylabel(r"$C_l$")
ax.set_xlabel(r"$\alpha$")
plt.legend(loc="best")
plt.show()
plt.style.use("ggplot")

clax1 = (cl0012_xfoil[1] - cl0012_xfoil[0]) / (3 * pi / 180)
print(clax1)

# Verification of new Linear Vortex Panel Method code
naca_code = "naca0012"
alpha = 8
solution = LinearVortex(
    airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
).solve_for(alpha=alpha)

xfoil_result = xfoil.find_pressure_coefficients(naca_code, alpha, delete=True)
fig, ax = solution.plot_pressure_distribution()
ax.scatter(
    xfoil_result["x"],
    xfoil_result["Cp"],
    marker=".",
    color="black",
    label="XFOIL",
)
ax.legend(loc="best")
