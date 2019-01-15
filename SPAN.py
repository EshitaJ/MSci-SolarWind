import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.integrate as spi
from utils import *


class SPAN:

    def __init__(self,
                 v_A,
                 T_x, T_y, T_z,
                 n, core_fraction,
                 bulk_velocity=700000
                 ):
        self.v_A = v_A
        self.T_x = T_x
        self.T_y = T_y
        self.T_z = T_z
        self.n = n
        self.core_fraction = core_fraction
        self.bulk_velocity = bulk_velocity

        self.latest_current_matrix = None
        self.latest_count_matrix = None

    def integral(self, vz, vx, vy):
        vdf = BiMax(x=vx, y=vy, z=vz,
                    v_A=self.v_A,
                    T_x=self.T_x, T_y=self.T_y, T_z=self.T_z,
                    n=self.n, core_fraction=self.core_fraction, is_core=True,
                    bulk_velocity=self.bulk_velocity) \
              + \
              BiMax(x=vx, y=vy, z=vz,
                    v_A=self.v_A,
                    T_x=self.T_x, T_y=self.T_y, T_z=self.T_z,
                    n=self.n, core_fraction=self.core_fraction, is_core=False,
                    bulk_velocity=self.bulk_velocity)
        return vdf

    def current_integral(self, vz, vx, vy):
        return self.integral(vz=vz, vx=vx, vy=vy) * cst.e * np.sqrt(vx**2 + vy**2 + vz**2)

    def pixel_current_integral(self, vz, vx, vy, theta_low, theta_high, phi_low, phi_high):

        if vx == 0. and vy == 0. and vz == 0.:
            return 0.

        vel = np.array([vx, vy, vz])
        v = np.linalg.norm(vel)

        """theta = vy/v
        phi = vx/v"""

        """theta = np.arccos(vy/v)
        phi = np.arccos(-vx/v)"""

        n = np.array([0.94, 0.35, 0])
        p = np.array([0.136, -0.367, -0.92])
        theta = np.arccos(1/np.linalg.norm(vel - np.dot(vel, p) * p))
        phi = np.arccos(1/np.linalg.norm(vel - np.dot(vel, n) * n))

        if theta_low < theta < theta_high and phi_low < phi < phi_high:
            return self.current_integral(vx=vx, vy=vy, vz=vz)

        else:
            return 0.

    def current_integrate(self, z_low, z_high, xy_lim, theta_low, theta_high, phi_low, phi_high):
        I_k = spi.tplquad(self.pixel_current_integral,
                          -xy_lim, xy_lim,
                          lambda x: -xy_lim, lambda x: xy_lim,
                          lambda x, y: z_low, lambda x, y: z_high,
                          # epsrel=1e-3, epsabs=0,
                          args=(theta_low, theta_high, phi_low, phi_high)
                          )[0]
        # print(I_k)
        return I_k

    def current_measure(self, z_low, z_high, xy_lim):
        theta_arr_deg = np.linspace(-60, 60, 33)
        theta_arr = theta_arr_deg*np.pi/180
        low_theta = theta_arr[:-1]
        high_theta = theta_arr[1:]

        phi_arr_deg = np.concatenate((np.linspace(0, 112.5, 11), np.linspace(135, 247.5, 6)))
        phi_arr = phi_arr_deg*np.pi/180
        low_phi = phi_arr[:-1]
        high_phi = phi_arr[1:]

        current_matrix = np.empty([32, 16])

        for theta_index in range(32):
            for phi_index in range(16):
                print(theta_index, phi_index)
                # print(low_theta[theta_index], high_theta[theta_index])
                # print(low_phi[phi_index], high_phi[phi_index])
                value = self.current_integrate(
                        z_low=z_low, z_high=z_high, xy_lim=xy_lim,
                        theta_low=low_theta[theta_index],
                        theta_high=high_theta[theta_index],
                        phi_low=low_phi[phi_index],
                        phi_high=high_phi[phi_index])
                current_matrix[theta_index][phi_index] = value

        self.latest_current_matrix = current_matrix

        print(current_matrix[current_matrix != 0.0].min(), current_matrix[current_matrix != 0.0].max())

        ax = plt.gca()

        plt.imshow(current_matrix, interpolation='none'
                   , norm=colors.LogNorm(vmin=current_matrix[current_matrix != 0.0].min(),
                                         vmax=current_matrix[current_matrix != 0.0].max())
                   )

        # ax.xaxis.set_xticks(phi_arr)
        # ax.xaxis.set_yticks(theta_arr)

        plt.xticks(phi_arr)
        plt.yticks(theta_arr)

        plt.show()

    def count_integral(self, v, theta, phi, eff_A=0.01, dt=0.001):
        x, y, z = sph_to_cart(r=v, theta=theta, phi=phi)
        vdf = self.integral(vz=z, vx=x, vy=y)
        jacobian = np.square(v) * np.cos(theta)

        dN = vdf * v * eff_A * dt * jacobian

        return dN

    def count_integrate(self, v_low, v_high, theta_low, theta_high, phi_low, phi_high):
        N = spi.tplquad(self.count_integral,
                        phi_low, phi_high,
                        lambda x: theta_low, lambda x: theta_high,
                        lambda x, y: v_low, lambda x, y: v_high)[0]

        return N

    def count_measure(self, v_low, v_high):
        theta_arr_deg = np.linspace(-60, 60, 33)
        theta_arr = theta_arr_deg * np.pi / 180
        low_theta = theta_arr[:-1]
        high_theta = theta_arr[1:]

        phi_arr_deg = np.concatenate((np.linspace(0, 112.5, 11), np.linspace(135, 247.5, 6)))
        phi_arr = phi_arr_deg * np.pi / 180
        low_phi = phi_arr[:-1]
        high_phi = phi_arr[1:]

        count_matrix = np.empty([32, 16])

        for theta_index in range(32):
            for phi_index in range(16):
                print(theta_index, phi_index)
                value = self.count_integrate(
                    v_low=v_low, v_high=v_high,
                    theta_low=low_theta[theta_index],
                    theta_high=high_theta[theta_index],
                    phi_low=low_phi[phi_index],
                    phi_high=high_phi[phi_index])
                count_matrix[theta_index][phi_index] = value

        self.latest_count_matrix = count_matrix

        plt.imshow(count_matrix, interpolation='none'
                   , norm=colors.LogNorm(vmin=count_matrix[count_matrix != 0.0].min(),
                                         vmax=count_matrix[count_matrix != 0.0].max())
                   )

        plt.show()


if __name__ == '__main__':
    z_l = 3.1e4
    z_h = 2.5e6
    xy = 3e6
    vA = 245531.8
    device = SPAN(v_A=vA, T_x=1e4, T_y=1e4, T_z=2e5, n=92e6, core_fraction=0.9)
    # device.current_measure(3.1e4, 2.5e6, 3e7, 0, 2*np.pi, 0, 2*np.pi)
    device.count_measure(v_low=z_l, v_high=z_h)

