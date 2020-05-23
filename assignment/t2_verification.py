"""Task 2: Verification with a thin NACA Airfoil."""

from matplotlib import pyplot as plt
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

from gammapy.panel.thick import ThickPanelledAirfoil, Solver
import numpy as np
import csv
from math import pi


def get_xfoil_cp(nacafoil):
    with open(r"reference_data/{}.txt".format(nacafoil),'r') as f:
        reader = csv.reader(f, skipinitialspace=True, delimiter=' ')
        Cp = np.array([[0, 0]])
        for i in range(3):
            next(reader)
        for row in reader:
            Cp = np.append(Cp, [ [float(row[0]), float(row[2])] ], axis=0)
    return np.delete(Cp, 0, 0)



def get_data(Naca='0012', alpha=0):
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


# * ### verification Cp ###
Cp1, cl1, foil1 = get_data(Naca='0012', alpha=3)
fig, ax = plt.subplots()
ax.plot([i[0] for i in foil1.panels.collocation_points], Cp1, "k", label="GammaPy")
ax.plot(*zip(*get_xfoil_cp('xfoil0012_a0')), "xr", markevery=5, label="XFOIL")
ax.invert_yaxis()
ax.set_ylabel(r"$C_p$")
ax.set_xlabel('x/c')
plt.legend(loc="best")
plt.show()
plt.style.use("ggplot")

# * ### verification Cl ###

def get_xfoil_cl(file):
    with open(r"reference_data/{}".format(file),'r') as f:
        reader = csv.reader(f, skipinitialspace=True, delimiter=' ')
        Cl = np.array([[0, 0]])
        for i in range(12):
            next(reader)
        for row in reader:
            Cl = np.append(Cl, [ [float(row[0]), float(row[1])] ], axis=0)
    return np.delete(Cl, 0, 0)

def get_cls(naca):
    alfas = [0,3,5,8,10,13,15]
    cl = []
    for a in alfas:
        _, Cl, _ = get_data(Naca=naca, alpha=a)
        cl.append(Cl)
    return np.reshape(cl, (len(alfas), 1))


cl0012 = get_cls('0012')
cl4412 = get_cls('4412')

cl0012_xfoil = get_xfoil_cl('Polar_naca0012_0_15')
cl4412_xfoil = get_xfoil_cl('Polar_naca4412_0_15')


fig, ax = plt.subplots()
ax.plot(*zip(*cl0012_xfoil), "k", label=" Xfoil")
ax.plot([i[0] for i in cl0012_xfoil], cl0012, "b", label=" GammaPy")
# ax.plot(*zip(*cl4412_xfoil), "g", label="NACA4412 Xfoil")
# ax.plot([i[0] for i in cl4412_xfoil], cl4412, "r", label="NACA4412 GammaPy")
ax.set_ylabel(r"$C_l$")
ax.set_xlabel(r'$\alpha$')
plt.legend(loc="best")
plt.show()
plt.style.use("ggplot")

fig, ax = plt.subplots()
ax.plot(*zip(*cl4412_xfoil), "g", label=" Xfoil")
ax.plot([i[0] for i in cl4412_xfoil], cl4412, "r", label=" GammaPy")
ax.set_ylabel(r"$C_l$")
ax.set_xlabel(r'$\alpha$')
plt.legend(loc="best")
plt.show()
plt.style.use("ggplot")

clax1 = cl0012_xfoil[-5][1]/(cl0012_xfoil[-5][0]/180*pi)
print(clax1)


