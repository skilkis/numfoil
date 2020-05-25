"""Task 5: Show difference between a thick/thin airfoil using XFOIL."""

from matplotlib import pyplot as plt

from gammapy.legacy.panel.thick import Solver, ThickPanelledAirfoil


def get_data(Naca="0012", alpha=0):
    foil = ThickPanelledAirfoil(Naca=Naca, n_panels=100)
    # foil.panels.plt()
    solver = Solver(foil.panels)
    Cp = solver.solve_Cp(alpha=alpha, plot=False)
    Cl = solver.get_cl(alpha=alpha)
    return Cp, Cl, foil


# * ### effect of thickness on Cp ###

# fig, ax = plt.subplots()

# for t in range(6, 30, 4):
#     Cp, _, foil = get_data(Naca='00{0:0=2d}'.format(t), alpha=5)
#     ax.plot([i[0] for i in foil.panels.collocation_points], Cp, label="NACA00{0:0=2d}".format(t))
# ax.invert_yaxis()
# ax.set_ylabel(r"$C_p$")
# ax.set_xlabel('x/c')
# plt.legend(loc="best")
# plt.show()
# plt.style.use("ggplot")


# * ### effect of thickness on CL ###
fig, ax = plt.subplots()
for a in [0, 5, 10, 15]:
    Cl = []
    t = []
    for i in range(6, 30, 4):
        _, cl, _ = get_data(Naca="44{0:0=2d}".format(i), alpha=a)
        Cl.append(cl)
        t.append(i)

    ax.plot(t, Cl, label="alpha = {}deg".format(a))
ax.set_ylabel(r"$C_l$")
ax.set_xlabel("thickness [%]")
plt.legend(loc="best")

plt.show()
plt.style.use("ggplot")
