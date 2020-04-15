import numpy as np
from math import pi, sin, cos, atan, atan2
from functions import airfoil, panel_points
import matplotlib.pyplot as plt

# pg280 573 300

# * Constants * #
alpha = 10 * pi/180
v_inf = 1.
u_inf = v_inf * cos(alpha)
w_inf = v_inf * sin(alpha)
rho = 1.225
q = 0.5 * rho * v_inf**2

n_panels = 5           # Number of panels
N = n_panels + 1        # Number of panel edges

NACA = [9, 4, 2, 0]     # NACA XXXX digits of 4-digit airfoil
t_c = 0.14              # airfoil thickness

# * obtaining panel edge, collocation and vortex points * #
coor_u, coor_l, coor_c = airfoil(NACA, N)
coor_col, coor_vor = panel_points(coor_c)





# Get all x- or y-values from a coordinate set for easier plotting
# i = 0 for x values, i = 1 for y values
def pt(set, i):
    p = []
    for point in set:
        p.append(point[i])
    return p

plt.plot(
        #  pt(coor_u, 0), pt(coor_u, 1), 'b',
        #  pt(coor_l, 0), pt(coor_l, 1), 'b',
         pt(coor_c, 0), pt(coor_c, 1), 'k',
         pt(coor_vor, 0), pt(coor_vor, 1), 'ob',
         pt(coor_col, 0), pt(coor_col, 1), 'xr')
plt.axis('equal')
plt.show()
