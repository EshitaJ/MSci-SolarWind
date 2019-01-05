import os
import numpy as np
import seaborn as sns
import scipy.integrate as spi
import scipy.constants as cst
from SPC_Fits import *
from Global_Variables import *
from SPC_Plates import *
from Quaternion import *
from VDF import *
sns.set()


def current_vdensity(vz, vy, vx, v, is_core, n):
    # df = BiMax(vz, vy, vx, v, is_core, n)
    df = rotatedMW(vz, vy, vx, v, is_core, n, B0)
    return cst.e * np.sqrt(vz**2 + vy**2 + vx**2) * Area(vz, vy, vx) * df


def Signal_Count(bounds, is_core):
    output = []
    low = bounds[:-1]
    high = bounds[1:]
    for i in range(len(bounds)-1):
        I_k = spi.tplquad(current_vdensity,
                          -lim, lim,
                          lambda x: -lim, lambda x: lim,
                          lambda x, y: low[i], lambda x, y: high[i],
                          args=(va, is_core, constants["n"]))
        output.append(I_k[0])
    return np.array(output)


def Data(velocities, is_core):
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


def Plot(E_plot, plot_total, is_core,
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

    xlabel = "{x}".format(x="Energy (ev)" if E_plot else "$V_z$ (km/s)")
    plt.xlabel(xlabel)
    plt.ylabel("Current (nA)")
    plt.legend()
