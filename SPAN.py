import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.integrate as spi
import scipy.optimize as spo
from decimal import Decimal
from utils import *


class SPAN:

    def __init__(self,
                 v_A,
                 T_par, T_perp,
                 n, core_fraction,
                 bulk_velocity_int=700000,
                 bulk_velocity_arr=np.array([-700000, 0, 0])
                 ):
        self.v_A = v_A
        self.T_par = T_par
        self.T_perp = T_perp
        self.n = n
        self.core_fraction = core_fraction
        self.bulk_velocity_int = bulk_velocity_int
        self.bulk_velocity_arr = bulk_velocity_arr

        self.latest_current_matrix = None
        self.latest_count_matrix = None
        self.latest_coarse_count_matrix = None

    def integral(self, vz, vx, vy):
        vdf = BiMax(x=vx, y=vy, z=vz,
                    v_A=self.v_A,
                    T_par=self.T_par, T_perp=self.T_perp,
                    n=self.n, core_fraction=self.core_fraction, is_core=True,
                    bulk_velocity=self.bulk_velocity_int) \
              + \
              BiMax(x=vx, y=vy, z=vz,
                    v_A=self.v_A,
                    T_par=self.T_par, T_perp=self.T_perp,
                    n=self.n, core_fraction=self.core_fraction, is_core=False,
                    bulk_velocity=self.bulk_velocity_int)
        return vdf

    def rotated_integral(self, vz, vx, vy):
        vdf = RotatedBiMaW(x=vx, y=vy, z=vz,
                           v_A=self.v_A,
                           T_par=self.T_par, T_perp=self.T_perp,
                           n=self.n, core_fraction=self.core_fraction, is_core=True,
                           bulk_velocity=self.bulk_velocity_arr) \
              + \
              RotatedBiMaW(x=vx, y=vy, z=vz,
                           v_A=self.v_A,
                           T_par=self.T_par, T_perp=self.T_perp,
                           n=self.n, core_fraction=self.core_fraction, is_core=False,
                           bulk_velocity=self.bulk_velocity_arr)
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

            #phi_arr_deg = np.concatenate((np.linspace(0, 112.5, 11), np.linspace(135, 247.5, 6)))
            phi_arr_deg = np.concatenate((np.linspace(0, 135, 7), np.linspace(146.25, 247.5, 10)))
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

        if mode == 'default':
            self.latest_count_matrix = count_matrix
        elif mode == 'coarse':
            self.latest_coarse_count_matrix = count_matrix

        if save_data:
            np.savetxt('SPANDataBulk-700B010n00-1m-100Tpar%.1ETperp%.1ECF%.0f.csv'
                       % (self.T_par, self.T_perp, self.core_fraction*10),
                          count_matrix, delimiter=',')

        else:
            plt.imshow(count_matrix, interpolation='none'
                       , norm=colors.LogNorm(vmin=count_matrix[count_matrix != 0.0].min(),
                                             vmax=count_matrix[count_matrix != 0.0].max())
                       )

            plt.title('SPAN Results\n'
                      'n = [0, 0.35, 0.94], m = [-0.39, -0.86, 0.32]')
            plt.xlabel('Phi Cell Index')
            plt.ylabel('Theta Cell Index')

            plt.show()

    def load_data(self, file_loc):
        data = np.genfromtxt(file_loc, delimiter=',')
        #print(data)
        self.latest_count_matrix = data

    def plot_data(self, mode='default', savefig=False, saveloc=None):

        norm_min = 1
        norm_max = 1e8

        f = plt.figure()
        ax = f.add_axes([0.1, 0.1, 0.72, 0.79])
        if mode == 'default':
            im = plt.imshow(self.latest_count_matrix, interpolation='none'
                            , norm=colors.LogNorm(vmin=norm_min,
                                                  vmax=norm_max)
                            )
        elif mode == 'coarse':
            im = plt.imshow(self.latest_coarse_count_matrix, interpolation='none'
                            , norm=colors.LogNorm(vmin=norm_min,
                                                  vmax=norm_max)
                            )
        else:
            print('choose a valid mode')

        plt.title('SPAN Results, 45$\degree$ Flow')
        plt.xlabel('Phi Cell Index')
        plt.ylabel('Theta Cell Index')

        plt.colorbar(im, ticks=np.geomspace(norm_min,
                                            norm_max,
                                            num=10),
                     format='$%.0E$')

        plt.legend(title="Bulk Speed = [%.0f, %.0f, %.0f]km/s\n"
                         "Tpar = %.0f, Tperp = %.0f\n"
                         "n=[0,1,0], m=[-1,0,0]\n"
                         "Field Direction=[-1,-1,0]"
                         % (self.bulk_velocity_arr[0] / 1e3, self.bulk_velocity_arr[1] / 1e3,
                            self.bulk_velocity_arr[2] / 1e3,
                            np.sqrt(cst.k * self.T_par / cst.m_p) / 1e3,
                            np.sqrt(2*cst.k * self.T_perp / cst.m_p) / 1e3),
                   loc='center right', bbox_to_anchor=(-0.3, 0.5))

        if savefig:
            plt.savefig(saveloc)

        plt.show()

    def theta_count_plot(self, v_low, v_high, theta_index, generate_own=True, ignore_SPAN_pos=False):
        phi_arr_deg = np.concatenate((np.linspace(0, 135, 7), np.linspace(146.25, 247.5, 10)))
        phi_arr = phi_arr_deg * np.pi / 180
        low_phi = phi_arr[:-1]
        high_phi = phi_arr[1:]
        mid_phi = (low_phi + high_phi)/(2*np.pi)

        theta_arr_deg = np.linspace(-60, 60, 33)
        theta_arr = theta_arr_deg * np.pi / 180
        low_theta = theta_arr[:-1]
        high_theta = theta_arr[1:]

        if theta_index == 'middle':
            theta_low_lim = low_theta[15]
            theta_high_lim = high_theta[16]

        else:
            theta_low_lim = low_theta[theta_index]
            theta_high_lim = high_theta[theta_index]

        if generate_own:
            count_array = np.empty(len(low_phi))
            for phi_index in range(len(low_phi)):
                print(phi_index)
                value = self.count_integrate(
                    v_low=v_low, v_high=v_high,
                    theta_low=theta_low_lim,
                    theta_high=theta_high_lim,
                    phi_low=low_phi[phi_index],
                    phi_high=high_phi[phi_index],
                    ignore_SPAN_pos=ignore_SPAN_pos)
                count_array[phi_index] = value

        else:
            count_array = self.latest_count_matrix[theta_index]

        # print(mid_phi)
        # print(count_array)

        def MaxFit(x, coeff, mean, std, const):
            value = coeff * np.exp(-0.5*((x - mean)/std)**2) + const
            return value

        popt, pcov = spo.curve_fit(MaxFit, mid_phi[count_array != 0], np.log10(count_array[count_array != 0]),
                                   p0=(150, 0.803, 0.25, -136))

        fit_x = np.linspace(0, 1.4, 100)

        fit_count = MaxFit(fit_x, *popt)
        # print(pcov)
        # print(fit_count)

        fig, ax = plt.subplots()

        plt.plot(mid_phi[count_array != 0], np.log10(count_array[count_array != 0]), label='Data Points', marker='x')
        plt.plot(fit_x, fit_count, label='Fit')

        plt.title('Counts Vs Phi\n'
                  'Aberated Radial Flow')
        plt.xticks(np.linspace(0, 1.4, 8))
        plt.xlabel('Phi Mid-Value/pi')
        plt.ylabel('log10(count)')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plt.legend(title="%.2f < Theta < %.2f\n"
                         "Bulk Speed = [%.0f, %.0f, %.0f]km/s\n"
                         "Tpar = %.0f, Tperp = %.0f\n"
                         "n=[0,1,0], m=[-1,0,0]\n"
                         "Fitted Mean Phi = %.3f$\pi$+-%.3f$\pi$\n"
                         "Calculated Mean Phi = 0.803$\pi$"
                         % (theta_low_lim, theta_high_lim,
                            self.bulk_velocity_arr[0]/1e3, self.bulk_velocity_arr[1]/1e3, self.bulk_velocity_arr[2]/1e3,
                            np.sqrt(cst.k * self.T_par / cst.m_p) / 1e3,
                            np.sqrt(cst.k * self.T_perp / cst.m_p) / 1e3,
                            popt[1], np.sqrt(pcov[1][1])
                            ),
                   loc='lower left', bbox_to_anchor=(0, 0))

        plt.show()

    def phi_count_plot(self, v_low, v_high, phi_index, generate_own=True, ignore_SPAN_pos=False):
        phi_arr_deg = np.concatenate((np.linspace(0, 135, 7), np.linspace(146.25, 247.5, 10)))
        phi_arr = phi_arr_deg * np.pi / 180
        low_phi = phi_arr[:-1]
        high_phi = phi_arr[1:]

        theta_arr_deg = np.linspace(-60, 60, 33)
        theta_arr = theta_arr_deg * np.pi / 180
        low_theta = theta_arr[:-1]
        high_theta = theta_arr[1:]
        mid_theta = (low_theta + high_theta)/(2*np.pi)

        if phi_index == 'middle':
            phi_low_lim = low_phi[15]
            phi_high_lim = high_phi[16]

        else:
            phi_low_lim = low_phi[phi_index]
            phi_high_lim = high_phi[phi_index]

        if generate_own:
            count_array = np.empty(len(low_theta))
            for theta_index in range(len(low_theta)):
                print(theta_index)
                value = self.count_integrate(
                    v_low=v_low, v_high=v_high,
                    theta_low=low_theta[theta_index],
                    theta_high=high_theta[theta_index],
                    phi_low=phi_low_lim,
                    phi_high=phi_high_lim,
                    ignore_SPAN_pos=ignore_SPAN_pos)
                count_array[theta_index] = value

        else:
            count_array = self.latest_count_matrix[phi_index]

        # print(mid_phi)
        # print(count_array)

        def MaxFit(x, coeff, mean, std, const):
            value = coeff * np.exp(-0.5*((x - mean)/std)**2) + const
            return value

        popt, pcov = spo.curve_fit(MaxFit, mid_theta[count_array != 0], np.log10(count_array[count_array != 0]),
                                   p0=(1, -0.25, 0.25, 1))

        fit_x = np.linspace(-1/3, 1/3, 100)

        fit_count = MaxFit(fit_x, *popt)
        print(popt)
        print(pcov)
        # print(fit_count)

        fig, ax = plt.subplots()

        plt.plot(mid_theta[count_array != 0], np.log10(count_array[count_array != 0]), label='Data Points', marker='x')
        plt.plot(fit_x, fit_count, label='Fit')

        plt.title('Counts Vs Theta\n'
                  'Aberated Radial Flow')
        plt.xticks(np.linspace(-0.4, 0.4, 9))
        plt.xlabel('Theta Mid-Value/pi')
        plt.ylabel('log10(count)')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plt.legend(title="%.2f < Phi < %.2f\n"
                         "Bulk Speed = [%.0f, %.0f, %.0f]km/s\n"
                         "Tpar = %.0f, Tperp = %.0f\n"
                         "n=[0,1,0], m=[-1,0,0]\n"
                         "Fitted Mean Theta = %.3f$\pi$+-%.3f$\pi$\n"
                         "Calculated Mean Theta = -0.250$\pi$"
                         % (phi_low_lim, phi_high_lim,
                            self.bulk_velocity_arr[0]/1e3, self.bulk_velocity_arr[1]/1e3, self.bulk_velocity_arr[2]/1e3,
                            np.sqrt(cst.k * self.T_par / cst.m_p) / 1e3,
                            np.sqrt(cst.k * self.T_perp / cst.m_p) / 1e3,
                            popt[1], np.sqrt(pcov[1][1])
                            ),
                   loc='lower left', bbox_to_anchor=(0, 0))

        plt.show()

    def pixel_energy_anlysis(self, theta_index, phi_index, resolution_number=100, find_FWHM=True, plot=True):
        theta_arr_deg = np.linspace(-60, 60, 33)
        theta_arr = theta_arr_deg * np.pi / 180
        low_theta_arr = theta_arr[:-1]
        high_theta_arr = theta_arr[1:]

        phi_arr_deg = np.concatenate((np.linspace(0, 135, 7), np.linspace(146.25, 247.5, 10)))
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

        if plot:
            plt.plot(v_mid_arr_km, count_array, marker='o', label='Measured Count')

        if find_FWHM:
            def BixMaxFit(x, mu1, mu2, std, coeff1, coeff2):
                return coeff1 * np.exp(-np.square(x - mu1) / (2 * np.square(std))) \
                       + coeff2 * np.exp(-np.square(x - mu2) / (2 * np.square(std)))

            #print(v_mid_arr, count_array)
            p, c = spo.curve_fit(BixMaxFit, v_mid_arr, count_array, p0=(700000, 750000, 37000, 1e7, 3e6))
            print(p)

            fitted_data = BixMaxFit(np.linspace(0, max(v_mid_arr), 1000), *p)
            if plot:
                plt.plot(np.linspace(0, max(v_mid_arr), 1000)/1e3, fitted_data,
                         label="Best double Gaussian fit with width %.3g %s"
                         % (p[2]/1e3, 'km/s'))

        if plot:
            plt.xlabel('Velocity/km')
            plt.ylabel('Count')

            plt.legend(title='T_par=%.0f km/s, T_perp=%.0f km/s\nResolution = %d'
                      % (np.sqrt(cst.k * self.T_par / cst.m_p)/1e3,
                         np.sqrt(cst.k * self.T_perp / cst.m_p)/1e3,
                         resolution_number))

            plt.title('Count Measured by Pixel Theta Index %d & Phi Index %d'
                      % (theta_index, phi_index))

            plt.savefig('PixelCountTpar%.1ETperp%.1ETheta%dPhi%dResolution%d.png'
                        % (self.T_par, self.T_perp, theta_index, phi_index, resolution_number))

            plt.show()

        if find_FWHM:
            return {'mu1': p[0], 'mu2': p[1], 'std': p[2], 'coeff1': p[3], 'coeff2': p[4]}, max(count_array)

    def temperature_search(self, resolution_number=100, B_extent=500000, noise_filter_fraction=0.01, direction='both',
                           mode='default', plot_brightest=False, plot_temperatures=True):
        if mode is 'default':
            if self.latest_count_matrix is not None:
                count_matrix = self.latest_count_matrix
            else:
                print('must generate data first!')
                count_matrix = None
        elif mode is 'coarse':
            if self.latest_coarse_count_matrix is not None:
                count_matrix = self.latest_coarse_count_matrix
            else:
                print('must generate data first!')
                count_matrix = None

        else:
            print('This is not a valid mode')
            count_matrix = None

        B_nm_x, B_nm_y, B_nm_z = nm_field_array

        phi_arr_deg = np.concatenate((np.linspace(0, 135, 7), np.linspace(146.25, 247.5, 10)))
        phi_arr = phi_arr_deg * np.pi / 180
        low_phi_arr = phi_arr[:-1]
        high_phi_arr = phi_arr[1:]
        mid_phi_arr = (low_phi_arr + high_phi_arr) / 2

        theta_arr_deg = np.linspace(-60, 60, 33)
        theta_arr = theta_arr_deg * np.pi / 180
        low_theta_arr = theta_arr[:-1]
        high_theta_arr = theta_arr[1:]
        mid_theta_arr = (low_theta_arr + high_theta_arr) / 2

        theta_max_index, phi_max_index = np.unravel_index(np.argmax(count_matrix, axis=None), count_matrix.shape)

        print(theta_max_index, phi_max_index)

        mid_phi_max = mid_phi_arr[phi_max_index]
        mid_theta_max = mid_theta_arr[theta_max_index]

        print(mid_theta_max, mid_phi_max)

        p_max, brightest_count = device.pixel_energy_anlysis(theta_index=theta_max_index, phi_index=phi_max_index,
                                                             resolution_number=resolution_number,
                                                             find_FWHM=True, plot=plot_brightest)

        print(brightest_count)

        bulk_x, bulk_y, bulk_z = sph_to_cart(p_max['mu1'], mid_theta_max, mid_phi_max)

        print(bulk_x, bulk_y, bulk_z)

        bulk_resolution = 2*B_extent + 1

        scale_array = np.linspace(-B_extent, B_extent, bulk_resolution)

        E_range_eV = np.linspace(5, 30e3, resolution_number + 1)
        E_range_J = E_range_eV * cst.e
        v_range = np.sqrt(2 * E_range_J / cst.m_p)
        low_v_arr = v_range[:-1]
        high_v_arr = v_range[1:]
        v_mid_arr = (low_v_arr + high_v_arr) / 2
        v_mid_arr_km = v_mid_arr / 1e3

        def BixMaxFit(x, mu1, mu2, std, coeff1, coeff2):
            return coeff1 * np.exp(-np.square(x - mu1) / (2 * np.square(std))) \
                   + coeff2 * np.exp(-np.square(x - mu2) / (2 * np.square(std)))

        colours = ['b', 'y', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                   'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        if direction is 'both' or 'parallel':
            parallel_x = scale_array*B_nm_x
            parallel_y = scale_array*B_nm_y
            parallel_z = scale_array*B_nm_z
            total_flow = np.transpose(np.array([parallel_x + bulk_x, parallel_y + bulk_y, parallel_z + bulk_z]))
            total_flow_sph = np.array([cart_to_sph_arr(total_flow[i]) for i in range(bulk_resolution)])
            total_flow_sph[:, 2][total_flow_sph[:, 2] < 0] += 2*np.pi

            flow_theta_phi_indices = np.empty([bulk_resolution, 2])

            for point_index in range(bulk_resolution):
                theta_index = next((i for i in range(32) if
                                   (total_flow_sph[point_index][1] >= low_theta_arr[i])
                                   & (total_flow_sph[point_index][1] <= high_theta_arr[i])), None)
                phi_index = next((i for i in range(16) if
                                 (total_flow_sph[point_index][2] >= low_phi_arr[i])
                                 & (total_flow_sph[point_index][2] <= high_phi_arr[i])), None)

                flow_theta_phi_indices[point_index] = np.array([theta_index, phi_index])

            unique_cells_nan = np.unique(flow_theta_phi_indices, axis=0)
            unique_cells = unique_cells_nan[~np.isnan(unique_cells_nan).any(axis=1)]
            print(unique_cells)

            cell_range = range(len(unique_cells[:, 0]))

            df_list = []

            for cell_index in cell_range:

                cell = unique_cells[cell_index]
                print(cell)
                theta = int(cell[0])
                phi = int(cell[1])

                colour_index = cell_index
                while colour_index > 11:
                    colour_index -= 12

                count_array = np.empty([resolution_number])

                for n in range(resolution_number):
                    print(n)
                    value = self.count_integrate(v_low=low_v_arr[n],
                                                 v_high=high_v_arr[n],
                                                 theta_low=low_theta_arr[theta],
                                                 theta_high=high_theta_arr[theta],
                                                 phi_low=low_phi_arr[phi],
                                                 phi_high=high_phi_arr[phi])
                    count_array[n] = value

                if not all(count_array < noise_filter_fraction*brightest_count):
                    if plot_temperatures:
                        plt.plot(v_mid_arr_km[count_array != 0], count_array[count_array != 0],
                                 marker='o', label='Measured: Theta %i, Phi %i' % (theta, phi),
                                 alpha=0.5, color=colours[colour_index])

                    p, c = spo.curve_fit(BixMaxFit, v_mid_arr, count_array,
                                         p0=(700000, 750000, 37000, max(count_array), max(count_array)/50),
                                         maxfev=10000)
                    print(p)

                    if plot_temperatures:
                        fitted_data = BixMaxFit(np.linspace(0, max(v_mid_arr), 1000), *p)
                        plt.plot(np.linspace(0, max(v_mid_arr), 1000)[fitted_data != 0]/1e3,
                                 fitted_data[fitted_data != 0],
                                 label="Fitted: Theta %i, Phi %i, with width %.3g %s"
                                 % (theta, phi, p[2]/1e3, 'km/s'), alpha=0.5, color=colours[colour_index])

                    cell_dict = {}
                    cell_dict.update({'theta': theta})
                    cell_dict.update({'phi': phi})
                    cell_dict.update({'max_count': max(count_array)})
                    cell_dict.update({'width': p[2]/1e3})
                    cell_dict.update({'mu1': p[0]})
                    if max(count_array) == brightest_count:
                        cell_dict.update({'brightest_pixel': True})
                    else:
                        cell_dict.update({'brightest_pixel': False})

                    df_list.append(cell_dict)

            if plot_temperatures:
                plt.xlabel('Velocty, km/s')
                plt.ylabel('Count')

                plt.legend(loc='upper right')

            cell_df = pd.DataFrame(df_list)
            print(cell_df)

            other_pixels_df = cell_df[~cell_df['brightest_pixel']].reset_index(drop=True)

            print(other_pixels_df)

            closest_index = None
            closest_max_count = 1e10

            for i in range(len(other_pixels_df.index)):
                if abs(other_pixels_df.iloc[i]['max_count'] - brightest_count/2) <\
                        abs(closest_max_count - brightest_count/2):
                    closest_index = i
                    closest_max_count = other_pixels_df.iloc[i]['max_count']
                    print(closest_max_count)

            print(closest_index, closest_max_count)

            closest_radius = other_pixels_df.iloc[closest_index]['mu1']
            closest_mid_theta = mid_theta_arr[int(other_pixels_df.iloc[closest_index]['theta'])]
            closest_mid_phi = mid_phi_arr[int(other_pixels_df.iloc[closest_index]['phi'])]
            print(closest_mid_theta, closest_mid_phi)

            closest_x, closest_y, closest_z = sph_to_cart(closest_radius, closest_mid_theta, closest_mid_phi)
            print(closest_x, closest_y, closest_z)
            parallel_temp_ms = 2*np.sqrt(np.square(bulk_x - closest_x) +
                                         np.square(bulk_y - closest_y) +
                                         np.square(bulk_z - closest_z))
            parallel_temp_k = cst.m_p * np.square(parallel_temp_ms) / cst.k
            print('parallel_temp = ', parallel_temp_ms, 'm/s, ',
                  parallel_temp_k, 'K')

        if direction is 'both' or 'perpendicular':
            B_nm_x, B_nm_y, B_nm_z = np.cross(np.array([B_nm_x, B_nm_y, B_nm_z]),
                                              np.array([bulk_x, bulk_y, bulk_z]) /
                                              np.linalg.norm(np.array([bulk_x, bulk_y, bulk_z])))
            print(B_nm_x, B_nm_y, B_nm_z)

            parallel_x = scale_array * B_nm_x
            parallel_y = scale_array * B_nm_y
            parallel_z = scale_array * B_nm_z
            total_flow = np.transpose(np.array([parallel_x + bulk_x, parallel_y + bulk_y, parallel_z + bulk_z]))
            total_flow_sph = np.array([cart_to_sph_arr(total_flow[i]) for i in range(bulk_resolution)])
            total_flow_sph[:, 2][total_flow_sph[:, 2] < 0] += 2 * np.pi

            flow_theta_phi_indices = np.empty([bulk_resolution, 2])

            for point_index in range(bulk_resolution):
                theta_index = next((i for i in range(32) if
                                    (total_flow_sph[point_index][1] >= low_theta_arr[i])
                                    & (total_flow_sph[point_index][1] <= high_theta_arr[i])), None)
                phi_index = next((i for i in range(16) if
                                  (total_flow_sph[point_index][2] >= low_phi_arr[i])
                                  & (total_flow_sph[point_index][2] <= high_phi_arr[i])), None)

                flow_theta_phi_indices[point_index] = np.array([theta_index, phi_index])

            unique_cells_nan = np.unique(flow_theta_phi_indices, axis=0)
            unique_cells = unique_cells_nan[~np.isnan(unique_cells_nan).any(axis=1)]
            print(unique_cells)

            cell_range = range(len(unique_cells[:, 0]))

            df_list = []

            for cell_index in cell_range:

                cell = unique_cells[cell_index]
                print(cell)
                theta = int(cell[0])
                phi = int(cell[1])

                colour_index = cell_index
                while colour_index > 11:
                    colour_index -= 12

                count_array = np.empty([resolution_number])

                for n in range(resolution_number):
                    print(n)
                    value = self.count_integrate(v_low=low_v_arr[n],
                                                 v_high=high_v_arr[n],
                                                 theta_low=low_theta_arr[theta],
                                                 theta_high=high_theta_arr[theta],
                                                 phi_low=low_phi_arr[phi],
                                                 phi_high=high_phi_arr[phi])
                    count_array[n] = value

                if not all(count_array < noise_filter_fraction * brightest_count):
                    if plot_temperatures:
                        plt.plot(v_mid_arr_km[count_array != 0], count_array[count_array != 0],
                                 marker='o', label='Measured: Theta %i, Phi %i' % (theta, phi),
                                 alpha=0.5, color=colours[colour_index])

                    p, c = spo.curve_fit(BixMaxFit, v_mid_arr, count_array,
                                         p0=(700000, 750000, 37000, max(count_array), max(count_array) / 50),
                                         maxfev=10000)
                    print(p)

                    if plot_temperatures:
                        fitted_data = BixMaxFit(np.linspace(0, max(v_mid_arr), 1000), *p)
                        plt.plot(np.linspace(0, max(v_mid_arr), 1000)[fitted_data != 0] / 1e3,
                                 fitted_data[fitted_data != 0],
                                 label="Fitted: Theta %i, Phi %i, with width %.3g %s"
                                       % (theta, phi, p[2] / 1e3, 'km/s'), alpha=0.5, color=colours[colour_index])

                    cell_dict = {}
                    cell_dict.update({'theta': theta})
                    cell_dict.update({'phi': phi})
                    cell_dict.update({'max_count': max(count_array)})
                    cell_dict.update({'width': p[2] / 1e3})
                    cell_dict.update({'mu1': p[0]})
                    if max(count_array) == brightest_count:
                        cell_dict.update({'brightest_pixel': True})
                    else:
                        cell_dict.update({'brightest_pixel': False})

                    df_list.append(cell_dict)

            if plot_temperatures:
                plt.xlabel('Velocty, km/s')
                plt.ylabel('Count')

                plt.legend(loc='upper right')

            cell_df = pd.DataFrame(df_list)
            print(cell_df)

            other_pixels_df = cell_df[~cell_df['brightest_pixel']].reset_index(drop=True)

            print(other_pixels_df)

            closest_index = None
            closest_max_count = 1e10

            for i in range(len(other_pixels_df.index)):
                if abs(other_pixels_df.iloc[i]['max_count'] - brightest_count / 2) < \
                        abs(closest_max_count - brightest_count / 2):
                    closest_index = i
                    closest_max_count = other_pixels_df.iloc[i]['max_count']
                    print(closest_max_count)

            print(closest_index, closest_max_count)

            closest_radius = other_pixels_df.iloc[closest_index]['mu1']
            closest_mid_theta = mid_theta_arr[int(other_pixels_df.iloc[closest_index]['theta'])]
            closest_mid_phi = mid_phi_arr[int(other_pixels_df.iloc[closest_index]['phi'])]
            print(closest_mid_theta, closest_mid_phi)

            closest_x, closest_y, closest_z = sph_to_cart(closest_radius, closest_mid_theta, closest_mid_phi)
            print(closest_x, closest_y, closest_z)
            perpendicular_temp_ms = 2 * np.sqrt(np.square(bulk_x - closest_x) +
                                                np.square(bulk_y - closest_y) +
                                                np.square(bulk_z - closest_z))
            peprendicular_temp_k = cst.m_p * np.square(perpendicular_temp_ms) / cst.k
            print('perpendicular = ', perpendicular_temp_ms , 'm/s, ',
                  peprendicular_temp_k, 'K')


if __name__ == '__main__':
    z_l = 3.09e4
    z_h = 1.38e6
    vA = 88000
    mode = 'default'
    device = SPAN(v_A=vA, T_par=170e3, T_perp=240e3, n=92e6, core_fraction=0.8,
                  bulk_velocity_arr=np.array([-700000, -700000, 0]))
    device.count_measure(v_low=z_l, v_high=z_h, mode=mode, ignore_SPAN_pos=False)
    #device.load_data('/home/henry/MSci-SolarWind/SPANDataBulk-700B010n010m00-1Tx1.7E+05Ty2.4E+05Tz2.4E+05CF8.csv')
    device.temperature_search(resolution_number=100, B_extent=500000, mode=mode, direction='both',
                              plot_brightest=False, plot_temperatures=True)
    device.plot_data(mode=mode, savefig=True,
                     saveloc='/home/henry/MSci-SolarWind/SPAN_Plots/Test.png')
