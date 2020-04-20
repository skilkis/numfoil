import numpy as np
from math import pi, sin, cos, atan, atan2, sqrt
from functions import airfoil, panels, v_ind, normals
import matplotlib.pyplot as plt

# pg280 573 300

# * Constants * #
alpha = 0 * pi/180
v_inf = 1.
u_inf = v_inf * cos(alpha)
w_inf = v_inf * sin(alpha)
Q_inf = np.array([[u_inf, w_inf]])
rho = 1.225
p_0 = 101325
# q = 0.5 * rho * v_inf**2

n_panels = 50            # Number of panels
NACA = [2, 4, 0, 6]     # NACA XXXX digits of 4-digit airfoil

coor_u, coor_l, coor_c = airfoil(NACA, n_panels)


def solve_vorticity(surface):
    coor_col, coor_vor, panel_angle, panel_length = panels(surface)
    normal, tangent = normals(panel_angle)

    A = np.zeros((n_panels, n_panels))
    RHS = np.zeros((n_panels, 1))

    for i, col in enumerate(coor_col):
        for j, vor in enumerate(coor_vor):
            q_ij = v_ind(col[0], col[1], vor[0], vor[1], 1)
            a_ij = np.dot(q_ij, normal[i])
            A[i, j] = a_ij
        RHS[i] = np.dot(-Q_inf, np.transpose(normal[i]))

    Gamma = np.linalg.solve(A, RHS)

    return Gamma, panel_length, coor_col


def solve_cp(Gamma, panel_length):
    dCp = dL = dP = np.zeros((n_panels, 1))

    for i, gamma in enumerate(Gamma):
        dL[i] = rho * v_inf * gamma
        dP[i] = rho * v_inf * gamma / panel_length[i]

        dCp[i] = (dP[i] - 101325)/(0.5*1.225*v_inf**2)

    return dL, dP, dCp

Gamma_u, panel_length_u, col_u = solve_vorticity(coor_u)
Gamma_l, panel_length_l, col_l = solve_vorticity(coor_l)

_, dP_u, dCp_u = solve_cp(Gamma_u, panel_length_u)
_, dP_l, dCp_l = solve_cp(Gamma_l, panel_length_l)

# * plotting * #

def pt(set, i):
    """
    Get all x- or y-values from a coordinate set for easier plotting.
    i = 0 for x values, i = 1 for y values

    Args:
        set (list): list of coordinates
        i (integer): Set i=0 for a list of x values, set i=1 for a list of all
        y vlaues

    Returns:
        [list]: list of all x- or all y-values from a list of coordinates
    """
    p = []
    for point in set:
        p.append(point[i])
    return p

# plt.figure(1)
# plt.plot(pt(coor_c, 0), pt(coor_c, 1), 'k',
#          pt(coor_vor, 0), pt(coor_vor, 1), 'ob',
#          pt(coor_col, 0), pt(coor_col, 1), 'xr',
#          pt(coor_u, 0), pt(coor_u, 1), 'b',
#          pt(coor_l, 0), pt(coor_l, 1), 'b',
#          )
# plt.axis('equal')
# plt.show()

plt.figure(2)
plt.plot(pt(col_u, 0), -dCp_u)
plt.plot(pt(col_l, 0), -dCp_l)
# plt.axis('equal')
plt.show()

# plt.figure(2)
# plt.plot(pt(coor_c, 0), -dCpu)
# plt.plot(pt(coor_c, 0), -dCpl)
# # plt.axis('equal')
# plt.show()