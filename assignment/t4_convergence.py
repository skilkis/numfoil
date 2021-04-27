"""Task 4: Convergence study on panel density."""

from matplotlib import pyplot as plt
from math import sqrt
from numfoil.legacy.panel.thick import Solver, ThickPanelledAirfoil

from pathlib import Path
from typing import Dict, Generator, List, Sequence

import numpy as np
import xfoil
from matplotlib import pyplot as plt

from numfoil.geometry import NACA4Airfoil
from numfoil.geometry.airfoil import ParabolicCamberAirfoil
from numfoil.solver.m_linear_vortex import LinearVortex, calc_linear_vortex_im
from numfoil.solver.m_lumped_vortex import LumpedVortex

DATA_DIR = Path(__file__).parent / "reference_data"
FIGURE_DIR = Path(__file__).parent.parent / "docs" / "static"


naca_code = "naca0015"
alpha = 5
# fig, ax = plt.subplots()
# for n_panels in [20, 60, 100]:
#     solution = LinearVortex(
#         airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=n_panels
#     ).solve_for(alpha=alpha)
#     fig, ax = solution.plot_pressure_distribution(figax=[fig,ax], label=f"{n_panels} panels")

#     # print(f"Lift Coefficient {naca_code} at {alpha} degrees is {solution.lift_coefficient}")
# xfoil_data = xfoil.find_pressure_coefficients(naca_code, alpha=alpha, delete=True)
# ax.plot(xfoil_data["x"], xfoil_data["Cp"], "xk", markevery=5, label="XFOIL")
# ax.legend()
# ax.legend(loc='best')
# fig.savefig(FIGURE_DIR / f"thick_panels.pdf", bbox_inches="tight")

# fig, ax = plt.subplots()
# for n_panels in [5, 20, 60]:
#     solution = LumpedVortex(
#         airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=n_panels
#     ).solve_for(alpha=alpha)
#     fig, ax = solution.plot_pressure_distribution(figax=[fig,ax], label=f"{n_panels} panels", stagp=False)

#     # print(f"Lift Coefficient {naca_code} at {alpha} degrees is
#     # {solution.lift_coefficient}")

# cps = xfoil.find_pressure_coefficients("naca0001", alpha=alpha, delete=True)['Cp']
# x = xfoil.find_pressure_coefficients("naca0001", alpha=alpha, delete=True)['x']
# mid = x.index(min(x))
# up = list(reversed(cps[mid:-1]))
# low = cps[:mid]
# ax.plot(x[:mid][::5], -np.array(up)[::5]+np.array(low)[::5], 'xk', label="XFOIL")

# ax.set_ylim([0, -5])
# ax.legend()
# ax.legend(loc='best')
# fig.savefig(FIGURE_DIR / f"thin_panels.pdf", bbox_inches="tight")





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



# fig, ax = plt.subplots()
# alpha = 5
# for naca_code in ["naca0015", "naca2415", "naca4415"]:
#     solution = LinearVortex(
#         airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
#     ).solve_for(alpha=alpha)
#     fig, ax = solution.plot_pressure_distribution(figax=[fig,ax], label=naca_code)
# # fig.savefig(FIGURE_DIR / "thick_camber_a5.pdf", bbox_inches="tight")

# fig, ax = plt.subplots()
# alpha = 5
# for naca_code in ["naca0015", "naca2415", "naca4415"]:
#     solution = LumpedVortex(
#         airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
#     ).solve_for(alpha=alpha)
#     fig, ax = solution.plot_delta_cp(figax=[fig,ax], label=naca_code)
#     ax.set_ylim([0,7])
# # fig.savefig(FIGURE_DIR / "thin_camber_a5.pdf", bbox_inches="tight")






# def get_data(Naca="0012", alpha=0, n_panels=100):
#     airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
#         ).solve_for(alpha=alpha)
#     solver = Solver(foil.panels)
#     Cp = solver.solve_Cp(alpha=alpha, plot=False)
#     Cl = solver.get_cl(alpha=alpha)
#     return Cp, Cl, foil


# ## * ### minimize dCl / run until convergence ###

for naca_code, alpha in [("naca0015", 5), ("naca2422", 10)]:

    mode="error"
    tol = 0.005
    itr = True
    p = 6
    t = []
    Cl = []

    # TODO # replace cl_old with xfoil value
    fig, ax = plt.subplots()

    solution = LinearVortex(
        airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=200
    ).solve_for(alpha=alpha)
    cl_old = solution.lift_coefficient[0][0]

    dCl = []
    extra = False

    ax.axhline(y=cl_old, color="k")
    ax.text(15, cl_old / 1.015, f"Goal $C_l = {round(cl_old,4)}$", color="k")
    # ax.set_zorder(1)

    while itr is True:
        solution = LinearVortex(
            airfoil=NACA4Airfoil(naca_code=naca_code, te_closed=True), n_panels=p
            ).solve_for(alpha=alpha)
        cl = solution.lift_coefficient[0][0]
        Cl.append(cl)
        t.append(p)

        diff = sqrt(((cl - cl_old) / cl_old) ** 2)
        dCl.append(diff)

        if diff < tol:
            if extra is True:
                itr = False
            else:
                ax.axvline(x=p, color="g")
                ax.text(
                    p * 0.65,
                    Cl[0] + (cl - Cl[0]) / 2,
                    "{} panels\n{} = {}\n$C_l$ = {}".format(
                        p, mode, round(diff, 4), round(cl, 4)
                    ),
                    color="g",
                )
                tol = tol * 0.5
                extra = True
                print(p, cl, cl_old, diff)
                print((cl - Cl[0]) / 2)
        else:
            p += 2
            if mode == "conv":
                cl_old = cl  # result conversion
            elif mode == "error":
                pass  # error reduction

    # print(p, cl, cl_old, diff)

    clplot = ax.plot(t, Cl)
    ax.set_ylabel(r"$C_l$", color=clplot[0].get_color())
    ax.set_xlabel(r"number of panels")
    ax.tick_params(axis="y", labelcolor=clplot[0].get_color())

    ax2 = ax.twinx()
    errorplot = ax2.plot(t, dCl, color=next(ax._get_lines.prop_cycler)['color'])
    ax2.set_ylabel(r"error $\epsilon $", color=errorplot[0].get_color())
    ax2.tick_params(axis="y", labelcolor=errorplot[0].get_color())

    ax2.grid(False)
    # fig.tight_layout()
    plt.show()
    plt.style.use("bmh")
    fig.savefig(FIGURE_DIR / f"thin_conv_{naca_code}.pdf", bbox_inches="tight")