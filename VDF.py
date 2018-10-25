import argparse
import plasmapy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as cst
from astropy import units as u
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab


constants = {
    "n": 92 * (10**6),  # m^-3
    "T_perp": 14e5,  # K
    "T_par": 3e5,  # K
    "B": 108e-9  # T
}


class VDF:
    """
    @TODO:
        sort out argparse for running VDFs
    """

    def __init__(
            self,
            n=constants["n"],
            T_perp=constants["T_perp"],
            T_par=constants["T_par"],
            B=constants["B"]):

        self.n = n
        self.T_perp = T_perp
        self.T_par = T_par
        self.B = B

        self.V_A = (plasmapy.physics.parameters.Alfven_speed(
                B*u.T,
                n*u.m**-3,
                ion="p+"))/(u.m/u.s)

    def VDF_2D_plot(
            self,
            core_fraction=0.8,
            meshgrid_points=600,
            v_perp_min=-1e1,
            v_perp_max=1e1,
            v_par_min=-1e6,
            v_par_max=1e6):

        vperp = np.linspace(v_perp_min, v_perp_max, meshgrid_points)
        vpar = np.linspace(v_par_min, v_par_max, meshgrid_points)

        x, y = np.meshgrid(vperp, vpar)

        core = self.BiMax_2D(x, y, 0, self.T_par, self.T_perp, core_fraction*self.n)
        beam = self.BiMax_2D(x, y, self.V_A, self.T_par, self.T_perp, (1-core_fraction)*self.n)

        z = core + beam

        C = plt.contour(x, y, z)
        # plt.rc('text', usetex=True)
        # plt.xlabel("$V_{\parallel}$   (m/s)", fontsize=20)
        # plt.ylabel("$V_{\perp}$   (m/s)", fontsize=20)
        # plt.clabel(C, inline = 1, fontsize = 10)

        plt.show()

    def VDF_3D_plot(
            self,
            core_fraction=0.8,
            meshgrid_points=200j,
            v_perp_min=-1e6,
            v_perp_max=1e6,
            v_par_min=-1e6,
            v_par_max=1e6):

        # Let z be the direction parallel ot the field,
        # and x, y perpendicular directions in plane perpendicular to the field
        """vperpx = np.linspace(v_perp_min, v_perp_max, meshgrid_points)
        vperpy = np.linspace(v_perp_min, v_perp_max, meshgrid_points)
        vparz = np.linspace(v_par_min, v_par_max, meshgrid_points)

        x, y, z = np.meshgrid(vperpx, vperpy, vparz)"""

        x, y, z = np.ogrid[v_perp_min:v_perp_max:meshgrid_points,
                           v_perp_min:v_perp_max:meshgrid_points,
                           v_par_min:v_par_max:meshgrid_points]

        core = self.BiMax_3D(z, x, y, 0, self.T_par, self.T_perp, core_fraction * self.n)
        beam = self.BiMax_3D(z, x, y, self.V_A, self.T_par, self.T_perp, (1 - core_fraction) * self.n)

        c = core + beam

        mlab.contour3d(c)

        mlab.show()

    @staticmethod
    def BiMax_2D(v_par, v_perp, drift_v, T_par, T_perp, n):
        par_thermal_speed = np.sqrt(2 * cst.k * T_par / cst.m_p)
        perp_thermal_speed = np.sqrt(2 * cst.k * T_perp / cst.m_p)

        bimax_value = n \
            / ((np.pi**1.5) * par_thermal_speed * np.square(perp_thermal_speed))\
            * np.exp(-np.square((v_par - drift_v) / par_thermal_speed))\
            * np.exp(-np.square(v_perp / perp_thermal_speed))

        return bimax_value

    @staticmethod
    def BiMax_3D(v_par, v_perp_x, v_perp_y, drift_v, T_par, T_perp, n):
        # Works on assumption of isotropy in plane perpendicular to field
        par_thermal_speed = np.sqrt(2 * cst.k * T_par / cst.m_p)
        perp_thermal_speed = np.sqrt(2 * cst.k * T_perp / cst.m_p)

        bimax_value = n \
            / ((np.pi ** 1.5) * par_thermal_speed * np.square(perp_thermal_speed)) \
            * np.exp(-np.square((v_par - drift_v) / par_thermal_speed)) \
            * np.exp(-np.square(v_perp_x / perp_thermal_speed)) \
            * np.exp(-np.square(v_perp_y / perp_thermal_speed))

        return bimax_value


"""parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, action=)"""

if __name__ == '__main__':
    test = VDF()
    test.VDF_3D_plot()