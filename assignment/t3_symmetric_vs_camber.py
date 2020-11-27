"""Task 3: Compare Symmetric vs. Asymmetric Airfoil."""


from matplotlib import pyplot as plt

from numfoil.legacy.panel.thick import Solver, ThickPanelledAirfoil


def get_data(Naca="0012", alpha=0):
    foil = ThickPanelledAirfoil(Naca=Naca, n_panels=100)
    # foil.panels.plt()
    solver = Solver(foil.panels)
    Cp = solver.solve_Cp(alpha=alpha, plot=False)
    Cl = solver.get_cl(alpha=alpha)
    return Cp, Cl, foil


# * ### effect of camber on Cp ###

fig, ax = plt.subplots()
Cp, _, foil = get_data(Naca="0012", alpha=5)
ax.plot([i[0] for i in foil.panels.collocation_points], Cp, label="NACA0012")
for c in range(1, 10, 2):
    Cp, cl, foil = get_data(Naca="{}412".format(c), alpha=5)
    ax.plot(
        [i[0] for i in foil.panels.collocation_points],
        Cp,
        label="NACA{}412".format(c),
    )
ax.invert_yaxis()
ax.set_ylabel(r"$C_p$")
ax.set_xlabel("x/c")
plt.legend(loc="best")
plt.show()
plt.style.use("ggplot")


# * ### effect of camber on CL ###
fig, ax = plt.subplots()
for a in [0, 5, 10, 15]:
    Cl = []
    c = []
    for i in range(10):
        _, cl, _ = get_data(Naca="2{}22".format(i), alpha=a)
        Cl.append(cl)
        c.append(i)

    ax.plot(c, Cl, label="alpha = {}deg".format(a))
ax.set_ylabel(r"$C_l$")
ax.set_xlabel("camber")
plt.legend(loc="best")

plt.show()
plt.style.use("ggplot")
