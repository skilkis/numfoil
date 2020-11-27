"""Task 4: Convergence study on panel density."""

from matplotlib import pyplot as plt
from math import sqrt
from numfoil.legacy.panel.thick import Solver, ThickPanelledAirfoil


def get_data(Naca="0012", alpha=0, n_panels=100):
    foil = ThickPanelledAirfoil(Naca=Naca, n_panels=n_panels)
    # foil.panels.plt()
    solver = Solver(foil.panels)
    Cp = solver.solve_Cp(alpha=alpha, plot=False)
    Cl = solver.get_cl(alpha=alpha)
    return Cp, Cl, foil


## * ### minimize dCl / run until convergence ###


def converge(tol=0.0005, mode="conv"):
    itr = True
    p = 6
    t = []
    Cl = []

    # TODO # replace cl_old with xfoil value
    _, cl_old, _ = get_data(Naca="0012", alpha=5, n_panels=200)
    print("goal = ", cl_old)
    dCl = []
    extra = False

    fig, ax = plt.subplots()
    ax.axhline(y=cl_old, color="k")
    ax.text(20, cl_old / 1.01, r"Goal $C_l$", color="k")
    ax.set_zorder(1)

    while itr is True:
        _, cl, _ = get_data(Naca="0012", alpha=5, n_panels=p)
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
                        p, mode, round(diff, 4), round(cl[0], 4)
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

    print(p, cl, cl_old, diff)

    ax.plot(t, Cl, "r")
    ax.set_ylabel(r"$C_l$", color="r")
    ax.set_xlabel(r"number of panels")
    ax.tick_params(axis="y", labelcolor="r")

    ax2 = ax.twinx()

    ax2.plot(t, dCl, "b")
    ax2.set_ylabel(r"$\Delta C_l$", color="b")
    ax2.tick_params(axis="y", labelcolor="b")

    fig.tight_layout()
    ax2.grid(False)

    plt.show()
    plt.style.use("ggplot")


converge(mode="error", tol=0.005)
# converge(mode='conv', tol=0.001)
