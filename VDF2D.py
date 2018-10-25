import argparse
import plasmapy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as cst
from astropy import units as u
from mpl_toolkits.mplot3d import Axes3D


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

    def BiMax(self, v_par, v_perp, v, n):
        par_thermal_speed = np.sqrt(2 * cst.k * self.T_par / cst.m_p)
        perp_thermal_speed = np.sqrt(2 * cst.k * self.T_perp / cst.m_p)

        bimax_value = n \
            / ((np.pi**1.5) * par_thermal_speed * np.square(perp_thermal_speed))\
            * np.exp(-np.square((v_par - v) / par_thermal_speed))\
            * np.exp(-np.square((v_perp) / perp_thermal_speed))

        return bimax_value

    def VDF_plot(
            self,
            core_fraction=0.8,
            meshgrid_points=600,
            v_perp_min=-1e6,
            v_perp_max=1e6,
            v_par_min=-1e6,
            v_par_max=1e6):

        vperp = np.linspace(v_perp_min, v_perp_max, meshgrid_points)
        vpar = np.linspace(v_par_min, v_par_max, meshgrid_points)

        x, y = np.meshgrid(vperp, vpar)
        print(core_fraction*self.n)
        print()
        print((1-core_fraction)*self.n)

        core = self.BiMax(x, y, 0, core_fraction*self.n)
        beam = self.BiMax(x, y, self.V_A, (1-core_fraction)*self.n)

        z = core + beam

        C = plt.contour(x, y, z)
        #plt.rc('text', usetex=True)
        #plt.xlabel("$V_{\parallel}$   (m/s)", fontsize=20)
        #plt.ylabel("$V_{\perp}$   (m/s)", fontsize=20)
        #plt.clabel(C, inline = 1, fontsize = 10)

        plt.show()


"""parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, action=)"""

if __name__ == '__main__':
    test = VDF()
    test.VDF_plot()