import os
import numpy as np
import seaborn as sns
import scipy.integrate as spi
import scipy.constants as cst
from SPC_Fits import *
sns.set()

constants = {
    "n": 92e6,  # m^-3
    "T_x": 14e5,  # K
    "T_y": 14e5,  # K
    "T_z": 3e5,  # K
    "B": 108e-9  # T
}

B = constants["B"]
B0 = np.array([B, 0, B])  # B field in SC frame
v_sc = np.array([0, 0, 20000])  # PSP velocity in m/s
print("B: ", B0)
va = np.linalg.norm(B0) / np.sqrt(cst.mu_0 * constants["n"] * cst.m_p)
print("V_alf: ", va)
lim = 1e8

"""SPC has an energy measurement range of 100 eV - 8 keV
corresponding to these velocities in m / s"""
J = 1.6e-19  # multiply for ev to J conversions
band_low = np.sqrt((2 * 100 * J) / cst.m_p)  # 138 km / s
band_high = np.sqrt((2 * 8e3 * J) / cst.m_p)  # 1237 km / s


def H(theta):
    """Function returning relative area on either side of the SPC;
    theta is angle from normal to SPC"""
    m = 1.51  # Value for SPC from alice's  report
    x = np.arccos(m * np.tan(theta))
    return (1 + (2/np.pi)*((np.sin(x)*np.cos(x)) - x))


def BiMax(z, y, x, v, is_core, n):
    """3D Bi-Maxwellian distribution;
    Well defined only in the B field frame, with main axis = B_z"""
    T_x = constants["T_x"]  # K
    T_y = constants["T_y"]
    T_z = constants["T_z"]
    core_fraction = 0.9

    if is_core:
        n_p = core_fraction * n
        # z_new = z - 700000
    else:
        n_p = (1 - core_fraction) * n
        # z_new = z - v - 700000

    norm = n_p * (cst.m_p/(2 * np.pi * cst.k))**1.5 / np.sqrt(T_x * T_y * T_z)
    exponent = -((z**2/T_z) + (y**2/T_y) + (x**2/T_x)) \
        * (cst.m_p/(2 * cst.k))
    vdf = norm * np.exp(exponent)
    return vdf


def Area(vz, vy, vx):
    """Effective area depends on the elevation_angle
    which is a function of velocity"""
    A_0 = 1.36e-4  # m^2
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    elevation_angle = np.degrees(np.arctan(vz / v) - np.pi/2)

    # can't have negative area
    A1 = 1 - (1.15 * elevation_angle) / 90
    A2 = 1 - (4.5 * (elevation_angle - 24))/90
    minimum = min(A1, A2)
    area = A_0 * minimum  # m^2
    return max(area, 0)


def rotationmatrix(B, z_axis):
    """Rotation matrix given according to Rodriugues' rotation formula
    for rotation by a given angle around a given axis;
    B and z are 3D row vectors;
    Either rotate B(VDF) onto z(SPC) or vice versa"""
    B = B / np.linalg.norm(B)
    z = z_axis / np.linalg.norm(z_axis)
    rot_vector = np.cross(B, z)
    # print(B, z, rot_vector)
    if np.linalg.norm(rot_vector) != 0:
        rot_axis = rot_vector / np.linalg.norm(rot_vector)
        cos_angle = np.dot(B, z)  # B and z are normalised
        print("Axis, angle: ", rot_axis, np.arccos(cos_angle))
        cross_axis = np.array(([0, -rot_axis[2], rot_axis[1]],
                              [rot_axis[2], 0, -rot_axis[0]],
                              [-rot_axis[1], rot_axis[0], 0]))
        outer_axis = np.outer(rot_axis, rot_axis)
        R = np.identity(3)*cos_angle + cross_axis*np.sqrt(1 - cos_angle**2) \
            + (1 - cos_angle)*outer_axis
    elif np.dot(B, z) > 0:
        # B and z parallel
        R = np.identity(3)
    else:
        # B and z anti-parallel
        R = -np.identity(3)
    # print(R)
    return R


def rotatedMW(vz, vy, vx, v, is_core, n, B):
    T_x = constants["T_x"]  # K
    T_y = constants["T_y"]
    T_z = constants["T_z"]
    core_fraction = 0.9

    R = rotationmatrix(B, np.array([0, 0, 1]))
    print(R)
    vel = np.array([vx, vy, vz])  # in SPC frame
    v_sw = np.array([0, 0, 700000])
    v_beam = np.array([0, 0, v])
    V = np.dot(R, vel)  # - v_sc
    print(V)
    V_SW = np.dot(R, v_sw)
    V_BEAM = np.dot(R, v_beam)

    if is_core:
        n_p = core_fraction * n
        v_new = V - V_SW
    else:
        n_p = (1 - core_fraction) * n
        v_new = V - V_SW - V_BEAM

    x, y, z = v_new

    norm = n_p * (cst.m_p/(2 * np.pi * cst.k))**1.5 / np.sqrt(T_x * T_y * T_z)
    exponent = -((z**2/T_z) + (y**2/T_y) + (x**2/T_x)) \
        * (cst.m_p/(2 * cst.k))
    DF = norm * np.exp(exponent)

    return DF


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
