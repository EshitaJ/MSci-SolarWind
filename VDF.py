import argparse
import plasmapy
import math
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.constants as cst
from astropy import units as u
# from mayavi import mlab
import rotate2


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
        consistency in labelling between rotate and here
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

        self.VDF_2D = None
        self.VDF_3D = None

        self.meshgrid_points_2D = None
        self.meshgrid_points_3D = None

        self.v_perp_min_3D = None
        self.v_perp_max_3D = None
        self.v_par_min_3D = None
        self.v_par_max_3D = None

        self.VDF_2D_generated = False
        self.VDF_3D_generated = False

    def gen_2D(self,
               core_fraction=0.8,
               meshgrid_points=600,
               v_perp_min=-1e6,
               v_perp_max=1e6,
               v_par_min=-1e6,
               v_par_max=1e6):

        # vperp = np.linspace(v_perp_min, v_perp_max, meshgrid_points)
        # vpar = np.linspace(v_par_min, v_par_max, meshgrid_points)

        x, y = np.mgrid[v_perp_min:v_perp_max:meshgrid_points, v_par_min:v_par_max:meshgrid_points]
        print(x, y)
        core = self.BiMax_2D(x, y, 0, self.T_par, self.T_perp, core_fraction*self.n)
        beam = self.BiMax_2D(x, y, self.V_A, self.T_par, self.T_perp, (1-core_fraction)*self.n)

        c = core + beam

        self.VDF_2D = (x, y, c)
        self.meshgrid_points_2D = meshgrid_points
        self.VDF_2D_generated = True

    def gen_3D(
            self,
            core_fraction=0.8,
            meshgrid_points=50,
            v_perp_min=-1e6,
            v_perp_max=1e6,
            v_par_min=-1e6,
            v_par_max=1e6):
        # Let z be the direction parallel ot the field,
        # and x, y perpendicular directions in plane perpendicular to the field

        self.meshgrid_points_3D = meshgrid_points
        meshgrid_points = meshgrid_points * 1j

        x, y, z = np.mgrid[v_perp_min:v_perp_max:meshgrid_points,
                           v_perp_min:v_perp_max:meshgrid_points,
                           v_par_min:v_par_max:meshgrid_points]

        self.v_perp_min_3D = v_perp_min
        self.v_perp_max_3D = v_perp_max
        self.v_par_min_3D = v_par_min
        self.v_par_max_3D = v_par_max

        # print(x, y, z)

        # print('x', x)
        # print('y', y)
        # print('z', z)

        # print('cheese')

        # print(np.stack([x, y, z], axis=3)[0, 0, 1, :])

        core = self.BiMax_3D(z, x, y, 0, self.T_par, self.T_perp, core_fraction * self.n)
        beam = self.BiMax_3D(z, x, y, self.V_A, self.T_par, self.T_perp, (1 - core_fraction) * self.n)

        c = core + beam

        # print(c)
        # print(x)
        # print(y)
        # print(z)

        y += 3e5

        self.VDF_3D = (x, y, z, c)
        self.VDF_3D_generated = True

    def plot_2D(self):

        assert self.VDF_2D_generated, "You need to generate the 2D VDF first via gen_2D()"

        x, y, c = self.VDF_2D

        C = plt.contour(x, y, c)
        # plt.rc('text', usetex=True)
        # plt.xlabel("$V_{\parallel}$   (m/s)", fontsize=20)
        # plt.ylabel("$V_{\perp}$   (m/s)", fontsize=20)
        # plt.clabel(C, inline = 1, fontsize = 10)

        plt.show()

    """
    def plot_3D(self):

        assert self.VDF_3D_generated, "You need to generate the 3D VDF first via gen_3D()"

        x, y, z, c = self.VDF_3D

        mlab.contour3d(x, y, z, c)

        mlab.show()
    """

    def plot_axis(self, axis, value):

        assert self.VDF_3D_generated, "You need to generate the 3D VDF first via gen_3D()"
        assert axis in ['x', 'y', 'z'], "This is not a valid axis to plot along"

        x, y, z, c = self.VDF_3D

        meshgrid_points = self.meshgrid_points_3D
        print(meshgrid_points)
        min_value = min(self.v_perp_min_3D, self.v_par_min_3D)
        max_value = max(self.v_perp_max_3D, self.v_par_max_3D)
        width = (max_value - min_value) / (meshgrid_points - 1)

        if axis == 'x':
            index = np.where((x > value - width/2) & (x <= value + width/2))
            ax = plt.axes(projection='3d')
            ax.plot_trisurf(y[index], z[index], c[index],
                            cmap='viridis', edgecolor='none')
            plt.xlabel('y')
            plt.ylabel('z')
            # plt.clabel(C, inline=1, fontsize=10)

        elif axis == 'y':
            index = np.where((y > value - width/2) & (y <= value + width/2))
            ax = plt.axes(projection='3d')
            ax.plot_trisurf(x[index], z[index], c[index],
                            cmap='viridis', edgecolor='none')
            plt.xlabel('x')
            plt.ylabel('z')

        else:
            index = np.where((z > value - width / 2) & (z <= value + width / 2))
            ax = plt.axes(projection='3d')
            ax.plot_trisurf(x[index], y[index], c[index],
                            cmap='viridis', edgecolor='none')
            plt.xlabel('x')
            plt.ylabel('y')

        plt.show()

    def B_ecliptic_transformation(self,
                                  phi,
                                  theta,
                                  psi=0,
                                  core_b_speed=None,
                                  core_ecliptic_speed=None):

        assert self.VDF_3D_generated, "You need to generate the 3D VDF first via gen_3D()"

        if core_ecliptic_speed is None:
            core_ecliptic_speed = np.array([0, 0, 0])

        if core_b_speed is None:
            core_b_speed = np.array([0, 0, 0])

        x, y, z, c = self.VDF_3D

        # x += core_b_speed[0]
        # y += core_b_speed[1]
        # z += core_b_speed[2]

        stack = np.stack([x, y, z], axis=3)

        new_x_axis, new_y_axis, new_z_axis = rotate2.b_ecliptic_coordinate_transform(
            phi=phi,
            theta=theta,
            psi=psi)

        for vx in range(int(abs(self.meshgrid_points_3D))):
            for vy in range(int(abs(self.meshgrid_points_3D))):
                for vz in range(int(abs(self.meshgrid_points_3D))):
                    # print(stack[vx, vy, vz, :])
                    # print(stack[vx, vy, vz, :])
                    # print(np.array(new_x_axis))
                    stack[vx, vy, vz, :] = np.array([np.dot(stack[vx, vy, vz, :], new_x_axis),
                                                     np.dot(stack[vx, vy, vz, :], new_y_axis),
                                                     np.dot(stack[vx, vy, vz, :], new_z_axis)])

                    # print(stack[vx, vy, vz, :])


        newx, newy, newz = np.split(stack, 3, axis=3)

        newx = np.squeeze(newx)
        newy = np.squeeze(newy)
        newz = np.squeeze(newz)

        newx += core_ecliptic_speed[0]
        newy += core_ecliptic_speed[1]
        newz += core_ecliptic_speed[2]

        self.VDF_3D = (newx, newy, newz, c)

    def find_3D(
            self,
            v_par,
            v_perp_x,
            v_perp_y,
            core_fraction=0.8,
            print_result=True):

        assert self.VDF_3D_generated, "You need to generate the 3D VDF first via gen_3D()"

        core = self.BiMax_3D(v_par, v_perp_x, v_perp_y,
                             0, self.T_par, self.T_perp,
                             core_fraction * self.n)
        beam = self.BiMax_3D(v_par, v_perp_x, v_perp_y,
                             self.V_A, self.T_par, self.T_perp,
                             (1 - core_fraction) * self.n)

        c = core + beam

        if print_result:
            print(f'The distribution is equal to {c}')

        return c


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
    meshgrid_points = 50
    test = VDF()
    # test.gen_2D(meshgrid_points=50j)
    # test.plot_2D()
    test.gen_3D(meshgrid_points=meshgrid_points, v_perp_min=-5e5, v_perp_max=5e5, v_par_min=-5e5, v_par_max=5e5)
    test.plot_axis(axis='x', value=0)
    test.plot_axis(axis='y', value=0)
    test.plot_axis(axis='z', value=0)
    test.B_ecliptic_transformation(phi=np.pi/2, theta=0, psi=0)
    test.plot_axis(axis='x', value=0)
    test.plot_axis(axis='y', value=0)
    test.plot_axis(axis='z', value=0)

