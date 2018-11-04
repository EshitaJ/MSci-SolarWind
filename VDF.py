import argparse
import plasmapy
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst
from astropy import units as u
from mayavi import mlab
import scipy.integrate as spi


constants = {
    "n": 92e6,  # m^-3
    "T_x": 3e5,  # K
    "T_y": 14e5,  # K
    "T_z": 14e5,
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
            # should split into separate T_beam and T_core
            T_x=constants["T_x"],
            T_y=constants["T_y"],
            B=constants["B"]):

        self.n = n
        self.T_x = T_x
        self.T_y = T_y
        self.B = B

        self.V_A = (plasmapy.physics.parameters.Alfven_speed(
                B*u.T,
                n*u.m**-3,
                ion="p+"))/(u.m/u.s)

    def VDF_2D_plot(
            self,
            core_fraction=0.8,
            meshgrid_points=500,
            v_perp_min=-1e6,
            v_perp_max=1e6,
            v_par_min=-1e6,
            v_par_max=1e6):

        vperp = np.linspace(v_perp_min, v_perp_max, meshgrid_points)
        vpar = np.linspace(v_par_min, v_par_max, meshgrid_points)

        x, y = np.meshgrid(vperp, vpar)

        core = self.BiMax_2D(x, y, 0, core_fraction*self.n)
        beam = self.BiMax_2D(x, y, self.V_A, (1-core_fraction)*self.n)

        z = core + beam

        # check for normalisation - this must return total n
        """lim = 1e8
        n1 = spi.dblquad(self.BiMax_2D, -lim, lim, -lim, lim,
                         args=(0, 0.8*self.n))
        n2 = spi.dblquad(self.BiMax_2D, -lim, lim, -lim, lim,
                         args=(0, 0.2*self.n))
        print(n1[0]+n2[0])"""  # should be same as n

        # Returns a 2D contour plot
        C = plt.contour(x, y, z)
        # plt.rc('text', usetex=True)
        # plt.xlabel("$V_{\parallel}$   (m/s)", fontsize=20)
        # plt.ylabel("$V_{\perp}$   (m/s)", fontsize=20)
        # plt.clabel(C, inline = 1, fontsize = 10)

        plt.show()

    def VDF_3D_plot(
            self,
            core_fraction=0.8,
            meshgrid_points=300j,
            # velocity is given in m / s
            v_perp_min=-1e5,
            v_perp_max=1e5,
            v_par_min=-1e5,
            v_par_max=1e5):

        # Let z be the direction parallel to the field,
        # and x, y perpendicular directions in plane perpendicular to the field

        x, y, z = np.ogrid[v_perp_min:v_perp_max:meshgrid_points,
                           v_perp_min:v_perp_max:meshgrid_points,
                           v_par_min:v_par_max:meshgrid_points]

        core = self.BiMax_3D(z, x, y, 0,
                             core_fraction * self.n)
        beam = self.BiMax_3D(z, x, y, self.V_A,
                             (1 - core_fraction) * self.n)

        c = core + beam

        # check for normalisation - this must return total n
        """lim = 1e6
        n1 = spi.tplquad(self.BiMax_3D, -lim, lim, -lim, lim, -lim, lim,
                         args=(0, core_fraction * self.n))
        n2 = spi.tplquad(self.BiMax_3D, -lim, lim, -lim, lim, -lim, lim,
                         args=(self.V_A, (1 - core_fraction) * self.n))
        print(n1[0]+n2[0])"""  # should be the same as n

        mlab.contour3d(c)
        # mlab.volume_slice(c)  # this would be cool with a slider bar
        mlab.show()

    @staticmethod
    def BiMax_2D(x, y, v, n):
        T_x = constants["T_x"]  # K
        T_y = constants["T_y"]
        norm = n * cst.m_p/(2 * np.pi * cst.k) / np.sqrt(T_x * T_y)
        exponent = -(((x-v)**2/T_x) + (y**2/T_y)) * (cst.m_p/(2 * cst.k))
        return norm * np.exp(exponent)

    @staticmethod
    def BiMax_3D(x, y, z, v, n):
        T_x = constants["T_x"]  # K
        T_y = constants["T_y"]
        T_z = constants["T_z"]
        norm = n * (cst.m_p/(2 * np.pi * cst.k))**1.5/np.sqrt(T_x * T_y * T_z)
        exponent = -(((x-v)**2/T_x) + (y**2/T_y) + (z**2/T_z)) \
            * (cst.m_p/(2 * cst.k))
        return norm * np.exp(exponent)


"""parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, action=)"""

if __name__ == '__main__':
    test = VDF()
    test.VDF_3D_plot()
