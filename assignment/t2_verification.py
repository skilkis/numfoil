"""Task 2: Verification with a thin NACA Airfoil."""

from matplotlib import pyplot as plt
from xfoil import find_pressure_coefficients

alpha = 1
values1 = find_pressure_coefficients("naca0016", alpha, delete=True)
values2 = find_pressure_coefficients("naca0012", alpha, delete=True)

plt.style.use("ggplot")
plot = plt.plot(values1["x"], values1["Cp"], label="NACA-0016")
plt.plot(values2["x"], values2["Cp"], label="NACA-0012")
plt.xlabel("Chord Fraction (x/c)")
plt.ylabel("Pressure Coefficient [-]")
plt.title(r"NACA Airfoils at $\alpha={alpha}$".format(alpha=alpha))
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.show()
