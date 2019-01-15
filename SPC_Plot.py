import os
import numpy as np
import seaborn as sns
import scipy.integrate as spi
import scipy.constants as cst
from SPC_Fits import *
from SPC_Plates import *
from Global_Variables import *
from SPC_Plates import *
from Quaternion import *
from VDF import *
sns.set()


def current_vdensity(vz, vy, vx, v, is_core, n):
    # df = BiMax(vz, vy, vx, v, is_core, n)
    df = rotatedMW(vz, vy, vx, v, is_core, n, B0)
    return cst.e * np.sqrt(vz**2 + vy**2 + vx**2) * Area(vz, vy, vx) * df


def integrand_plate(vz, vy, vx, v_alf, n, plate):
    """integral returns the current in a given detector plate"""

    v = np.sqrt(vx**2 + vy**2 + vz**2)
    th_x = -np.arcsin(vx/v)
    th_y = np.arcsin(vy/v)

    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    angular_range = np.pi / 6

    if -angular_range < th_x < angular_range \
            and -angular_range < th_y < angular_range:
        cvd = current_vdensity(vz, vy, vx, v_alf, True, n) \
            + current_vdensity(vz, vy, vx, v_alf, False, n)
        return cvd * Detector(th_x, th_y, plate)
    else:
        return 0


def Signal_Count(bounds, is_core, plates, plate):
    start1 = timeit.default_timer()
    output = []
    low = bounds[:-1]
    high = bounds[1:]

    def integration(low_bound, high_bound):
        # start = timeit.default_timer()
        if plates:
            I_k = spi.tplquad(integrand_plate,
                              -lim, lim,
                              lambda x: -lim, lambda x: lim,
                              lambda x, y: low_bound, lambda x, y: high_bound,
                              args=(va, constants["n"], plate))
            # stop = timeit.default_timer()
            # print("T: ", stop-start)
        else:
                I_k = spi.tplquad(current_vdensity,
                                  -lim, lim,
                                  lambda x: -lim, lambda x: lim,
                                  lambda x, y: low_bound,
                                  lambda x, y: high_bound,
                                  args=(va, is_core, constants["n"]))
                # stop = timeit.default_timer()
                # print("T: ", stop-start)
        return I_k

    integration = np.vectorize(integration)
    output = integration(low, high)[0]
    # stop1 = timeit.default_timer()
    # print("T1: ", stop1-start1)
    return output


def Data(velocities, is_core, plates, plate):
    if plates:
        if plate == 1:
            if os.path.isfile("quad_1_data.csv"):
                signal = np.genfromtxt("quad_1_data.csv")
            else:
                signal = Signal_Count(velocities, is_core, plates, plate) * 1e9

                np.savetxt("quad_1_data.csv", signal)
                print("Saved, quad 1 data")
        if plate == 2:
            if os.path.isfile("quad_2_data.csv"):
                signal = np.genfromtxt("quad_2_data.csv")
            else:
                signal = Signal_Count(velocities, is_core, plates, plate) * 1e9

                np.savetxt("quad_2_data.csv", signal)
                print("Saved, quad 2 data")
        if plate == 3:
            if os.path.isfile("quad_3_data.csv"):
                signal = np.genfromtxt("quad_3_data.csv")
            else:
                signal = Signal_Count(velocities, is_core, plates, plate) * 1e9

                np.savetxt("quad_3_data.csv", signal)
                print("Saved, quad 3 data")
        if plate == 4:
            if os.path.isfile("quad_4_data.csv"):
                signal = np.genfromtxt("quad_4_data.csv")
            else:
                signal = Signal_Count(velocities, is_core, plates, plate) * 1e9

                np.savetxt("quad_4_data.csv", signal)
                print("Saved, quad 4 data")

    else:
        if is_core:
            if os.path.isfile("core_signal_data.csv"):
                signal = np.genfromtxt("core_signal_data.csv")
            else:
                signal = Signal_Count(velocities, is_core) * 1e9

                np.savetxt("core_signal_data.csv", signal)
                print("Saved, core")
        else:
            if os.path.isfile("beam_signal_data.csv"):
                signal = np.genfromtxt("beam_signal_data.csv")
            else:
                signal = Signal_Count(velocities, is_core) * 1e9

                np.savetxt("beam_signal_data.csv", signal)
                print("Saved, beam")
    return signal


def Plot(E_plot, plot_total, is_core, plates,
         mu1_guess, mu2_guess, variance_guess, num=50):
    """ @ E_plot=True plots current against Energy, and otherwise plots
    current against z-velocity;
    @ plot_total=True plots the total core+beam vdf with a bi-max fit,
    and otherwise plots and fits either core or beam vdf
    (function needs to be called once for core and once for beam if =False);
    @ is_core=True (relevant iff plot_total=False) calculates and plots core,
    and otherwise beam;
    @ mu1_guess and mu2_guess are used as estimates of
    core and beam peaks respectively for fitting;
    @ variance_guess is estimate of variance for fitting (for now,
    assume core and beam distributions have same width)
    @ num is number of bins, default is set to 50
    """

    # unequal bin widths in velocity as potential is what is varied
    # for now assume equal bin widths in potential, but can change later
    potential = np.linspace(100, 8e3, int(num))
    vz_m = np.sqrt((2 * potential * J) / cst.m_p)  # z velocity in m/s

    vz_k = vz_m / 1e3  # velocity in km/s for plotting purposes
    v_band_centres = (vz_k[:-1] + vz_k[1:]) / 2.0
    fit_array_v = np.linspace(np.min(vz_k), np.max(vz_k), int(1e3))

    E_band_centres = (potential[:-1] + potential[1:]) / 2.0
    fit_array_E = np.linspace(np.min(potential), np.max(potential), int(1e3))

    if E_plot:
        band_centres = E_band_centres
        band_width = np.diff(potential)
        fit_array = fit_array_E
    else:
        band_centres = v_band_centres
        band_width = np.diff(vz_k)
        fit_array = fit_array_v

    if plates:
        quad1 = Data(vz_m, True, True, 1)
        quad2 = Data(vz_m, True, True, 2)
        quad3 = Data(vz_m, True, True, 3)
        quad4 = Data(vz_m, True, True, 4)
        total = quad1 + quad2 + quad3 + quad4
        Total_Fit(E_plot, band_centres, quad1/total, fit_array, 'Quadrant 1',
                  mu1_guess, mu2_guess, variance_guess)
        Total_Fit(E_plot, band_centres, quad2/total, fit_array, 'Quadrant 2',
                  mu1_guess, mu2_guess, variance_guess)
        Total_Fit(E_plot, band_centres, quad3/total, fit_array, 'Quadrant 3',
                  mu1_guess, mu2_guess, variance_guess)
        Total_Fit(E_plot, band_centres, quad4/total, fit_array, 'Quadrant 4',
                  mu1_guess, mu2_guess, variance_guess)
        # Total_Fit(E_plot, band_centres, total/total, fit_array, 'Total',
                  # mu1_guess, mu2_guess, variance_guess)
        plt.ylabel("Fractional Current")

    else:
        if plot_total:
            core = Data(vz_m, True)  # velocities in m/s for calculations
            beam = Data(vz_m, False)  # velocities in m/s for calculations
            total = core + beam
            # can either plot total or beam stacked on core - both are same
            plt.bar(band_centres, core, width=band_width,
                    label="Measured core at $T_z = %g$" % constants["T_z"])
            plt.bar(band_centres, beam, width=band_width, bottom=core,
                    label="Measured beam at $T_z = %g$" % constants["T_z"])
            Total_Fit(E_plot, band_centres, total, fit_array,
                      mu1_guess, mu2_guess, variance_guess)
        else:
            signal = Data(vz_m, is_core)  # velocities in m/s for calculations
            mu_guess = mu1_guess if is_core else mu2_guess
            FWHM(E_plot, is_core, band_centres, signal, fit_array,
                 mu_guess, variance_guess)
            plt.bar(band_centres, signal, width=band_width,
                    label="Measured %s at $T_z = %g$"
                    % ("core" if is_core else "beam", constants["T_z"]))
        plt.ylabel("Current (nA)")

    xlabel = "{x}".format(x="Energy (ev)" if E_plot else "$V_z$ (km/s)")
    plt.xlabel(xlabel)
    plt.legend(title="Signal in SPC with B field at "
               "(%g$^\\circ$, %g$^\\circ$) from the SPC normal"
               % (np.degrees(np.arctan(B0[0]/B0[2])),
                  np.degrees(np.arctan(B0[1]/B0[2]))))
