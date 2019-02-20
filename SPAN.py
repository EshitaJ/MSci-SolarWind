import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.integrate as spi
import scipy.optimize as spo
from decimal import Decimal
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

    def rotated_integral(self, vz, vx, vy):
        vdf = RotatedBiMaW(x=vx, y=vy, z=vz,
                           v_A=self.v_A,
                           T_x=self.T_x, T_y=self.T_y, T_z=self.T_z,
                           n=self.n, core_fraction=self.core_fraction, is_core=True,
                           bulk_velocity=np.array([0, self.bulk_velocity, 0])) \
              + \
              RotatedBiMaW(x=vx, y=vy, z=vz,
                           v_A=self.v_A,
                           T_x=self.T_x, T_y=self.T_y, T_z=self.T_z,
                           n=self.n, core_fraction=self.core_fraction, is_core=False,
                           bulk_velocity=np.array([0, self.bulk_velocity, 0]))
        return vdf

    """def current_integral(self, vz, vx, vy):
        return self.integral(vz=vz, vx=vx, vy=vy) * cst.e * np.sqrt(vx**2 + vy**2 + vz**2)

    def pixel_current_integral(self, vz, vx, vy, theta_low, theta_high, phi_low, phi_high):

        if vx == 0. and vy == 0. and vz == 0.:
            return 0.

        vel = np.array([vx, vy, vz])
        v = np.linalg.norm(vel)

        n = np.array([0.94, 0.35, 0])
        p = np.array([0.136, -0.367, -0.92])
        theta = np.arccos(1 / np.linalg.norm(vel - np.dot(vel, p) * p))
        phi = np.arccos(1 / np.linalg.norm(vel - np.dot(vel, n) * n))

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
    """

    def count_integral(self, v, theta, phi, eff_A=0.01, dt=0.001):
        x, y, z = sph_to_cart(r=v, theta=theta, phi=phi)
        vdf = self.integral(vz=z, vx=x, vy=y)
        jacobian = np.square(v) * np.cos(theta)

        dN = vdf * v * eff_A * dt * jacobian

        return dN

    def rotated_count_integral(self, v, theta, phi, eff_A=0.01, dt=0.001):
        x, y, z = sph_to_cart(r=v, theta=theta, phi=phi)
        vdf = self.rotated_integral(vz=z, vx=x, vy=y)
        jacobian = np.square(v) * np.cos(theta)

        dN = vdf * v * eff_A * dt * jacobian

        return dN

    def count_integrate(self, v_low, v_high, theta_low, theta_high, phi_low, phi_high, ignore_SPAN_pos=False):
        if ignore_SPAN_pos:
            N = spi.tplquad(self.count_integral,
                            phi_low, phi_high,
                            lambda x: theta_low, lambda x: theta_high,
                            lambda x, y: v_low, lambda x, y: v_high)[0]

        else:
            N = spi.tplquad(self.rotated_count_integral,
                            phi_low, phi_high,
                            lambda x: theta_low, lambda x: theta_high,
                            lambda x, y: v_low, lambda x, y: v_high)[0]

        return N

    def count_measure(self, v_low, v_high, save_data=True, ignore_SPAN_pos=False, mode='default'):
        if mode == 'default':
            theta_arr_deg = np.linspace(-60, 60, 33)
            theta_arr = theta_arr_deg * np.pi / 180
            low_theta = theta_arr[:-1]
            high_theta = theta_arr[1:]

            phi_arr_deg = np.concatenate((np.linspace(0, 112.5, 11), np.linspace(135, 247.5, 6)))
            phi_arr = phi_arr_deg * np.pi / 180
            low_phi = phi_arr[:-1]
            high_phi = phi_arr[1:]

        elif mode == 'coarse':
            theta_arr_deg = np.linspace(-60, 60, 6)
            theta_arr = theta_arr_deg * np.pi / 180
            low_theta = theta_arr[:-1]
            high_theta = theta_arr[1:]

            phi_arr_deg = np.linspace(0, 247.5, 9)
            phi_arr = phi_arr_deg * np.pi / 180
            low_phi = phi_arr[:-1]
            high_phi = phi_arr[1:]

        else:
            print('this is not a valid mode')

        count_matrix = np.empty([len(low_theta), len(low_phi)])

        for theta_index in range(len(low_theta)):
            for phi_index in range(len(low_phi)):
                print(theta_index, phi_index)
                value = self.count_integrate(
                    v_low=v_low, v_high=v_high,
                    theta_low=low_theta[theta_index],
                    theta_high=high_theta[theta_index],
                    phi_low=low_phi[phi_index],
                    phi_high=high_phi[phi_index],
                    ignore_SPAN_pos=ignore_SPAN_pos)
                count_matrix[theta_index][phi_index] = value

        self.latest_count_matrix = count_matrix

        if save_data:
            np.savetxt('SPANDataxRealRotTx%.1ETy%.1ETz%.1ECF%.0f.csv'
                       % (self.T_x, self.T_y, self.T_z, self.core_fraction*10),
                       count_matrix, delimiter=',')

        else:
            plt.imshow(count_matrix, interpolation='none'
                       , norm=colors.LogNorm(vmin=count_matrix[count_matrix != 0.0].min(),
                                             vmax=count_matrix[count_matrix != 0.0].max())
                       )

            plt.title('SPAN Results streaming along R \n T_R=%.0fkm/s, T_T=%.0fkm/s, T_N=%.0fkm/s \n'
                      'n = [0, 0.35, 0.94], m = [-0.39, -0.86, 0.32]'
                      % (self.T_x, self.T_y, self.T_z))
            plt.xlabel('Phi Cell Index')
            plt.ylabel('Theta Cell Index')

            plt.show()

    def load_data(self, file_loc):
        data = np.genfromtxt(file_loc, delimiter=',')
        #print(data)
        self.latest_count_matrix = data

    def plot_data(self, savefig=False, saveloc=None):

        norm_min = 1e-10
        norm_max = 1e8

        f = plt.figure()
        ax = f.add_axes([0.1, 0.1, 0.72, 0.79])
        im = plt.imshow(self.latest_count_matrix, interpolation='none'
                        , norm=colors.LogNorm(vmin=norm_min,
                                              vmax=norm_max)
                        )

        plt.title('SPAN Results streaming along y, Rotation First\n T_x=%.0fkm/s, T_y=%.0fkm/s, T_z=%.0fkm/s'
                  '\n n = [1, 0, 0], m = [0, 1, 0]'
                  % (np.sqrt(cst.k * self.T_x / cst.m_p) / 1e3,
                     np.sqrt(cst.k * self.T_y / cst.m_p) / 1e3,
                     np.sqrt(cst.k * self.T_z / cst.m_p) / 1e3))
        plt.xlabel('Phi Cell Index')
        plt.ylabel('Theta Cell Index')

        plt.colorbar(im, ticks=np.geomspace(norm_min,
                                            norm_max,
                                            num=10),
                     format='$%.0E$')

        if savefig:
            plt.savefig(saveloc)

        plt.show()

    def pixel_energy_anlysis(self, theta_index, phi_index, resolution_number=100, find_FWHM=True):
        theta_arr_deg = np.linspace(-60, 60, 33)
        theta_arr = theta_arr_deg * np.pi / 180
        low_theta_arr = theta_arr[:-1]
        high_theta_arr = theta_arr[1:]

        phi_arr_deg = np.concatenate((np.linspace(0, 112.5, 11), np.linspace(135, 247.5, 6)))
        phi_arr = phi_arr_deg * np.pi / 180
        low_phi_arr = phi_arr[:-1]
        high_phi_arr = phi_arr[1:]

        low_theta = low_theta_arr[theta_index]
        high_theta = high_theta_arr[theta_index]
        low_phi = low_phi_arr[phi_index]
        high_phi = high_phi_arr[phi_index]

        E_range_eV = np.linspace(5, 30e3, resolution_number+1)
        E_range_J = E_range_eV*cst.e

        E_mid_arr_eV =(E_range_eV[:-1] + E_range_eV[1:])/2
        E_mid_arr_J = E_mid_arr_eV*cst.e

        v_range = np.sqrt(2*E_range_J/cst.m_p)
        low_v_arr = v_range[:-1]
        high_v_arr = v_range[1:]
        v_mid_arr = (low_v_arr + high_v_arr)/2
        v_mid_arr_km = v_mid_arr/1e3

        count_array = np.empty([resolution_number])

        for n in range(resolution_number):
            print(n)
            value = self.count_integrate(v_low=low_v_arr[n],
                                         v_high=high_v_arr[n],
                                         theta_low=low_theta,
                                         theta_high=high_theta,
                                         phi_low=low_phi,
                                         phi_high=high_phi)
            count_array[n] = value

        fig, ax = plt.subplots()

        text_properties = dict(alpha=0.5)

        textstr = 'T_x=%.0f km/s, T_y=%.0f km/s, T_z=%.0f km/s \n Resolution = %d, Streaming along y'\
                  % (np.sqrt(cst.k * self.T_x / cst.m_p)/1e3,
                     np.sqrt(cst.k * self.T_y / cst.m_p)/1e3,
                     np.sqrt(cst.k * self.T_z / cst.m_p)/1e3,
                     resolution_number)

        ax.text(0.35, 0.95, textstr, transform=ax.transAxes, verticalalignment='top', bbox=text_properties)

        plt.plot(v_mid_arr_km, count_array, marker='o')

        if find_FWHM:
            def BixMaxFit(x, mu1, mu2, var, n):
                return n * np.exp(-(x - mu1) ** 2 / var) \
                       + (1-n) * np.exp(-(x - mu2) ** 2 / var)

            p, c = spo.curve_fit(BixMaxFit, v_mid_arr, count_array)

            fitted_data = BixMaxFit(v_mid_arr, *p)
            plt.plot(v_mid_arr_km, fitted_data,
                     label="Best double Gaussian fit with width %g %s"
                     % (p[3]**0.5, 'm/s'))

        plt.xlabel('Velocity/km')
        plt.ylabel('Count')

        plt.title('Count Measured by Pixel Theta Index %d & Phi Index %d'
                  % (theta_index, phi_index))

        plt.savefig('PixelCountyTx%.1ETy%.1ETz%.1ETheta%dPhi%dResolution%d.png'
                    % (self.T_x, self.T_y, self.T_z, theta_index, phi_index, resolution_number))

        plt.show()


if __name__ == '__main__':
    z_l = 3.1e4
    z_h = 4e6
    xy = 3e6
    vA = 245531.8
    device = SPAN(v_A=vA, T_x=1.4e6, T_y=3e5, T_z=1.4e6, n=92e6, core_fraction=0.9, bulk_velocity=700000)
    device.count_measure(v_low=z_l, v_high=z_h, mode='default', ignore_SPAN_pos=True)
    #device.load_data('/home/henry/MSci-SolarWind/Data/y_Stream/Bulk_Speed700km/SPANDatayTx1.4E+06Ty3.0E+05Tz1.4E+06CF9.csv')
    device.plot_data(savefig=True,
                     saveloc='/home/henry/MSci-SolarWind/SPAN_Plots/1.png')
    # device.pixel_energy_anlysis(theta_index=15, phi_index=7, resolution_number=100)
