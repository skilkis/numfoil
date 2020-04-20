import numpy as np
from math import pi, sin, cos, atan, sqrt

def airfoil(Naca=[2, 4, 1, 0], n_panels=10):
    """
    From a given 4-digit NACA code, finds the function of the camber line,
    thickness distribution, and upper and lower surface points

    Args:
        Naca (list, optional): list containing 4 digits of a NACA airfoil.
                               Defaults to [2, 4, 1, 0].
        n_panels (int, optional): Number of panels.
                           Defaults to 10.

    Returns:
        coor_u (list): List of coordinates of the upper airfoil surface.
        coor_l (list): List of coordinates of the lower airfoil surface.
        coor_c (list): List of coordinates of points on the camber line.
    """
    M = Naca[0]/100
    P = Naca[1]/10
    t_c = float("0.{}{}".format(Naca[2], Naca[3]))

    N = n_panels + 1        # Number of panel edges

    a0 = 0.2969
    a1 = -0.126
    a2 = -0.3516
    a3 = 0.2843
    a4 = -0.1015            # -0.1036 for closed TE, -0.1015 for open TE

    coor_u = []
    coor_l = []
    coor_c = []
    coor_t = []

    # using NACA 4 digit airfoil calculation to
    # obtain airfoil coordinate points = panel edges
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

        # exact position of upper and lower surface coordinates
        # perpendicular to camber line
        x_u = x - y_t * sin(theta)
        y_u = y_c + y_t * cos(theta)
        x_l = x + y_t * sin(theta)
        y_l = y_c - y_t * cos(theta)

        coor_u.append([x_u, y_u])   # upper airfoil coordinates
        coor_l.append([x_l, y_l])   # lower airfoil coordinates
        coor_t.append([x, y_t])     # thickness distribution
        coor_c.append([x, y_c])     # camber line coordinates = panel edges

    return coor_u, coor_l, coor_c


def panels(coor_c):
    """
    takes a list of points on the camber line. These points are considered
    panel edges. Collocation points are added at 0.75 panel length and vortex 
    points are added at 0.25 panel lenght.

    Args:
        coor_c (list): List of panel edges (on camber line).
                       should be list of camberline points.

    Returns:
        coor_col (list): list of coordinates of collocation points.
        coor_vor (list): list of coordinates of vortex element points.
    """
    coor_col = []
    coor_vor = []
    thetas = []
    panel_length = []

    for i in range(len(coor_c)-1):
        p1 = coor_c[i]
        p2 = coor_c[i+1]

        # place collocation point at 0.75 panel length
        x_col = p1[0] + 0.75 * (p2[0] - p1[0])  # x_i
        y_col = p1[1] + 0.75 * (p2[1] - p1[1])  # y_i
        coor_col.append([x_col, y_col])

        # place vortex point at 0.25 panel length
        x_vor = p1[0] + 0.25 * (p2[0] - p1[0])  # x_j
        y_vor = p1[1] + 0.25 * (p2[1] - p1[1])  # x_j
        coor_vor.append([x_vor, y_vor])

        # find panel angle
        theta = atan((p1[1] - p2[1]) / (p2[0] - p1[0]))
        thetas.append(theta)

        # panel length
        l = sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        panel_length.append(l)

    return coor_col, coor_vor, thetas, panel_length


def v_ind(x_col, y_col, x_vor, y_vor, Gamma=1):
    """
    Finds induced velocity at point x_col, y_col due to a vortex
    with circulation Gamma at x_vor, y_vor

    Args:
        x_col (Float): x location of evaluation point
        y_col (Float): y location of evaluation point
        x_vor (Float): x location of vortex element
        y_vor (Float): y location of vortex element
        Gamma (Float): Vorticity. Defaults to unit strength 1.

    Returns:
        u (Float): horizontal component of induced velocity due to vortex
        y (Float): vertical component of induced velocity due to vortex
    """

    r = sqrt( (x_col - x_vor)**2 + (y_col - y_vor)**2 )
    u = Gamma/(2*pi) * (y_col - y_vor)/r**2         # horizontal component
    w = -Gamma/(2*pi) * (x_col - x_vor)/r**2        # vertical component
    return np.array([u, w])


def normals(thetas):
    """
    Return the normal and tangent vectors of each panel from panel angels

    Args:
        thetas (list): List of panel angels

    Returns:
        normals (list): list of numpy arrays containing the normal vectors for
                        all panels
        tangents (list): list of numpy arrays containing the tangent vectors
                         for all panels
    """
    normals = []
    tangents = []
    for theta in thetas:
        n_i = np.array([sin(theta), cos(theta)])    # normal vector
        t_i = np.array([cos(theta), -sin(theta)])   # tangent vector
        normals.append(n_i)
        tangents.append(t_i)
    return normals, tangents
