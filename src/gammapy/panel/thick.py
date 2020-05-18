
from math import pi, sin, cos, tan, atan, atan2
import numpy as np
from gammapy.thick_functions import airfoil, normals, make_panels, v_comp, v_ind_loc
import matplotlib.pyplot as plt
import pylab as pl

# chow pg 160



class ThickPanelledAirfoil:

    def __init__(self, Naca='2422', n_panels=10):
        self.n_panels = n_panels            # Number of panels per surface
        self._naca = None
        self.panels = None
        self.NACA = Naca   # NACA XXXX digits of 4-digit airfoil

    @property
    def NACA(self):
        return self._naca

    @NACA.setter
    def NACA(self, value):
        self._naca = list(map(int, value))
        self.panels = panels(self._naca, self.n_panels)
        
    
    def plt(self):
        upper_surface, lower_surface, coor_c = airfoil(self.NACA, self.n_panels)
        plt.plot(
                *zip(*coor_c), "k", 
                *zip(*upper_surface), "b", 
                *zip(*lower_surface), "b",
                )
        plt.axis("equal")
        plt.show()


class panels:

    def __init__(self, NACA, n_panels):
        self.n_panels = n_panels
        self.get_panel_nodes(NACA)
        self.collocation_points, self.vortex_points, self.panel_angles, self.panel_lengths = make_panels(self.panel_nodes)
        self.normals, self.tangents = normals(self.panel_angles)
    
    def get_panel_nodes(self, NACA):
        coor_u, coor_l, _ = airfoil(NACA, self.n_panels)
        coor = list(coor_l)
        coor.reverse()
        coor.pop(-1)
        coor += coor_u
        self.panel_nodes = coor

    def plt(self, num=False, points=False):
        plt.plot(*zip(*self.panel_nodes), "k")

        if points is True:
            plt.plot(*zip(*self.vortex_points), 'x',
                     *zip(*self.collocation_points), 'o')

            if num is True:
                self.plt_numbers(self.collocation_points, 'blue')
                self.plt_numbers(self.vortex_points, 'red')
        plt.axis("equal")
        plt.show()
    
    def plt_numbers(self, lst, clr):
        i = 0
        for x, y in lst:
            pl.text(x, y, str(i), color=clr, fontsize=12)
            i += 1
            pl.margins(0.1)


class Solver:
    def __init__(self, panels):
        self.panels = panels
        self.coefficients = Coefficients(self.panels)


    def solve_Cp(self, alpha=8, plot=True):
        alpha = alpha * pi / 180
        Gamma = self.solve_vorticity(alpha)

        Cp = np.zeros((self.panels.n_panels, 1))
        v_ind = self.get_v_ind(alpha, Gamma)

        for i in range(self.panels.n_panels):           
            Cp[i] = 1 - v_ind[i]**2
        
        if plot is True:
            x = [i[0] for i in self.panels.collocation_points]
            
            plt.figure(3)
            plt.plot(x, Cp, "k")
            # plt.gca().set_ylim([-5,1])
            plt.gca().invert_yaxis()
            plt.show()

        return Cp
    
    def get_dCp(self, alpha=8, plot=True):
        Cp = self.solve_Cp(alpha)
        mid = int(len(Cp)/2)
        Cp_u = Cp[mid:]
        Cp_l = Cp[:mid]
        Cp_l = np.flip(Cp_l)
        dCp = Cp_u - Cp_l

        if plot is True:
            x = [i[0] for i in self.panels.collocation_points[mid:]]
            plt.figure(4)
            plt.plot(x, dCp, "k")
            # plt.gca().set_ylim([-5,1])
            plt.gca().invert_yaxis()
            plt.show()
        
        return Cp_u, Cp_l

    


    def solve_vorticity(self, alpha):
        Gamma = np.linalg.solve(self.coefficients.AN, self.get_RHS(alpha))
        return Gamma
    
    def get_RHS(self, alpha):
        RHS = np.zeros((self.panels.n_panels+1, 1))
        for i in range(self.panels.n_panels):
            RHS[i] = sin(self.panels.panel_angles[i] - alpha)
        RHS[self.panels.n_panels] = 0
        return RHS

    def get_v_ind(self, alpha, Gamma):
        v_ind = np.zeros((self.panels.n_panels, 1))

        for i in range(self.panels.n_panels):
            v_ind[i] = cos(self.panels.panel_angles[i] - alpha)
            for j in range(self.panels.n_panels+1):
                v_ind[i] = v_ind[i] + self.coefficients.AT[i][j] * Gamma[j]
        
        return v_ind


    
class Coefficients:

    def __init__(self, panels):
        self.panels = panels
        empty_matrix_C = np.zeros((self.panels.n_panels, self.panels.n_panels))
        empty_matrix_A = np.zeros((self.panels.n_panels +1, self.panels.n_panels +1))
        self.CN1, self.CN2, self.CT1, self.CT2 = map(np.copy, [empty_matrix_C] * 4)
        self.AN, self.AT = map(np.copy, [empty_matrix_A] * 2)

        for i in range(self.panels.n_panels):
            for j in range(self.panels.n_panels):

                if i == j :
                    self.CN1[i][j] = -1
                    self.CN2[i][j] = 1
                    self.CT1[i][j] = pi/2
                    self.CT2[i][j] = pi/2
                
                else:
                    self.A = ( 
                        - (self.panels.collocation_points[i][0]-self.panels.panel_nodes[j][0]) * cos(self.panels.panel_angles[j])
                        - (self.panels.collocation_points[i][1]-self.panels.panel_nodes[j][1]) * sin(self.panels.panel_angles[j])
                    )
                    
                    self.B = (
                        (self.panels.collocation_points[i][0]-self.panels.panel_nodes[j][0])**2
                        + (self.panels.collocation_points[i][1]-self.panels.panel_nodes[j][1])**2
                    )

                    self.C = (
                        sin(self.panels.panel_angles[i]-self.panels.panel_angles[j])
                    )

                    self.D = (
                        cos(self.panels.panel_angles[i]-self.panels.panel_angles[j])
                    )

                    self.E = (
                        (self.panels.collocation_points[i][0]-self.panels.panel_nodes[j][0]) * sin(self.panels.panel_angles[j])
                        - (self.panels.collocation_points[i][1]-self.panels.panel_nodes[j][1]) * cos(self.panels.panel_angles[j])
                    )

                    self.F = (
                        np.log( 1 + self.panels.panel_lengths[j] * (
                            self.panels.panel_lengths[j] + 2 * self.A ) / self.B )
                    )

                    self.G = (
                        atan2( self.E * self.panels.panel_lengths[j],
                                self.B + self.A * self.panels.panel_lengths[j])
                    )

                    self.P = (
                        (self.panels.collocation_points[i][0]-self.panels.panel_nodes[j][0])
                        * sin( self.panels.panel_angles[i] - 2 * self.panels.panel_angles[j])
                        +
                        (self.panels.collocation_points[i][1]-self.panels.panel_nodes[j][1])
                        * cos( self.panels.panel_angles[i] - 2 * self.panels.panel_angles[j])
                    )

                    self.Q = (
                        (self.panels.collocation_points[i][0]-self.panels.panel_nodes[j][0])
                        * cos( self.panels.panel_angles[i] - 2 * self.panels.panel_angles[j])
                        -
                        (self.panels.collocation_points[i][1]-self.panels.panel_nodes[j][1])
                        * sin( self.panels.panel_angles[i] - 2 * self.panels.panel_angles[j])
                    )

                    self.CN2[i][j] = ( 
                                self.D + 0.5 * self.Q * self.F / self.panels.panel_lengths[j]
                                - (self.A * self.C + self.D * self.E) * self.G / self.panels.panel_lengths[j]
                                )

                    self.CN1[i][j] = 0.5 * self.D * self.F + self.C * self.G - self.CN2[i][j]

                    self.CT2[i][j] = (
                                self.C + 0.5 * self.P * self.F / self.panels.panel_lengths[j]
                                + (self.A * self.D - self.C * self.E) * self.G / self.panels.panel_lengths[j]
                                )

                    self.CT1[i][j] = 0.5 * self.C * self.F - self.D * self.G - self.CT2[i][j]

        
        for i in range(self.panels.n_panels):
            self.AN[i][0] = self.CN1[i][0]
            self.AN[i][self.panels.n_panels] = self.CN2[i][self.panels.n_panels-1]
            self.AT[i][0] = self.CT1[i][0]
            self.AT[i][self.panels.n_panels] = self.CT2[i][self.panels.n_panels-1]
            for j in range(1, self.panels.n_panels):
                self.AN[i][j] = self.CN1[i][j] + self.CN2[i][j-1]
                self.AT[i][j] = self.CT1[i][j] + self.CT2[i][j-1]
        
        self.AN[self.panels.n_panels][:] = 0
        self.AN[self.panels.n_panels][0] = 1
        self.AN[self.panels.n_panels][self.panels.n_panels] = 1




if __name__ == '__main__':
    foil = ThickPanelledAirfoil(Naca='2412', n_panels=100)
    foil.panels.plt()
    solver = Solver(foil.panels)
    Cp = solver.solve_Cp(alpha=8, plot=True)
    u, l = solver.get_dCp(Cp, plot=True)