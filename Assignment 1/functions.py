import numpy as np
from math import pi, sin, cos, atan


def airfoil(Naca=[2, 4, 1, 0], N=101):

    M = Naca[0]/100
    P = Naca[1]/10
    t_c = float("0.{}{}".format(Naca[2], Naca[3]))

    a0 = 0.2969
    a1 = -0.126
    a2 = -0.3516
    a3 = 0.2843
    a4 = -0.1015        # -0.1036 for closed TE, -0.1015 for open TE

    coor_u = []
    coor_l = []
    coor_c = []
    coor_t = []

    # using NACA 4 digit airfoil calculation to
    # obtain airfoil coordinate points
    for beta in np.linspace(0, pi, N):
        # cosine spacing
        x = 0.5 - 0.5 * cos(beta)
        # thickness distribution
        y_t = t_c/0.2 * (a0 * x**0.5 +
                         a1 * x +
                         a2 * x**2 +
                         a3 * x**3 +
                         a4 * x**4)

        if x < P:
            # camber line before t_max
            y_c = M / P**2 * (2*P*x - x**2)
            # gradient before t_max
            dyc_dx = 2*M / P**2 * (P - x)

        elif x >= P:
            # camber line after t_max
            y_c = M / (1 - P)**2 * (1 - 2*P + 2*P*x - x**2)
            # gradient after t_max
            dyc_dx = 2*M / (1 - P)**2 * (P - x)

        theta = atan(dyc_dx)

        # exact position perpendicular to camber line
        x_u = x - y_t * sin(theta)
        y_u = y_c + y_t * cos(theta)
        x_l = x + y_t * sin(theta)
        y_l = y_c - y_t * cos(theta)

        coor_u.append([x_u, y_u])   # upper airfoil coordinates
        coor_l.append([x_l, y_l])   # lower airfoil coordinates
        coor_c.append([x, y_c])     # camber line coordinates
        coor_t.append([x, y_t])     # thickness distribution

    return coor_u, coor_l, coor_c


def panel_points(coor):

    coor_col = []
    coor_vor = []

    for i in range(len(coor)-1):
        p1 = coor[i]
        p2 = coor[i+1]

        # place collocation point at 0.75 panel length
        x_col = p1[0] + 0.75 * (p2[0] - p1[0])  # x_i
        y_col = p1[1] + 0.75 * (p2[1] - p1[1])  # y_i
        coor_col.append([x_col, y_col])

        # place vortex point at 0.25 panel length
        x_vor = p1[0] + 0.25 * (p2[0] - p1[0])  # x_j
        y_vor = p1[1] + 0.25 * (p2[1] - p1[1])  # x_j
        coor_vor.append([x_vor, y_vor])

    return coor_col, coor_vor