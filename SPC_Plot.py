import os
import numpy as np
import seaborn as sns
import peakutils
import scipy.optimize as spo
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.constants as cst
sns.set()

constants = {
    "n": 92e6,  # m^-3
    "T_x": 14e5,  # K
    "T_y": 14e5,  # K
    "T_z": 3e5,  # K
    "B": 108e-9  # T
}

B = constants["B"]
B0 = np.array([0, 0, B])  # B field in SC frame
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
        z_new = z - 700000
    else:
        n_p = (1 - core_fraction) * n
        z_new = z - v - 700000

    norm = n_p * (cst.m_p/(2 * np.pi * cst.k))**1.5 / np.sqrt(T_x * T_y * T_z)
    exponent = -((z_new**2/T_z) + (y**2/T_y) + (x**2/T_x)) \
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
        # print("Axis, angle: ", rot_axis, cos_angle)
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
    R = rotationmatrix(B, np.array([0, 0, 1]))
    vel = np.array([vx, vy, vz])  # in SPC frame
    Vx, Vy, Vz = np.dot(R, vel) - v_sc  # in B frame, where VDF is well-defined
    DF = BiMax(Vz, Vy, Vx, v, is_core, n)  # -V because 'look angle'
    return DF


def current_vdensity(vz, vy, vx, v, is_core, n):
    df = BiMax(vz, vy, vx, v, is_core, n)
    # df = rotatedMW(vz, vy, vx, v, is_core, n, B0)
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


def FWHM(E_plot, is_core, x_axis, data, fit_array, mu_guess, variance_guess):
    """Get FWHM of the data"""

    def fit_func(x, variance, mu, N):
        """1D maxwellian fit"""
        if E_plot:
            return N \
                * np.exp(-(np.sqrt(x) - np.sqrt(mu))**2 / variance)
        else:
            return N * np.exp(-(x - mu)**2 / variance)

    p, c = spo.curve_fit(fit_func, x_axis, data,
                         p0=(variance_guess, mu_guess, 0.09))

    if E_plot:

        def new_fit(x):
            return fit_func(x, *p) - (fit_func(p[1], *p) / 2.)

        print(*p)
        x1 = 2000 if is_core else 4000  # estimates of zeros for core and beam
        x2 = 3000 if is_core else 5000
        zeros = spo.root(new_fit, [x1, x2])
        print(zeros.x)
        fwhm = zeros.x[1] - zeros.x[0]
        # fwhm = (2 * p[0] * np.log(2))**2

    else:

        fwhm = 2 * np.sqrt(p[0] * np.log(2))

    plt.plot(fit_array, fit_func(fit_array, *p),
             label="Best Gaussian %s fit (FWHM = %g %s)"
             % ("core" if is_core else "beam",
             fwhm, "eV" if E_plot else "km/s"))
    return fwhm


def Total_Fit(E_plot, x_axis, data, fit_array,
              mu1_guess, mu2_guess, variance_guess):
    """Get a fit of total data"""

    def fit_func(x, variance, mu1, mu2, N1, N2):
        """Bi-maxwellian fit"""
        if E_plot:
            x_new = np.sqrt(x)
            mu1_new = np.sqrt(mu1)
            mu2_new = np.sqrt(mu2)
        else:
            x_new = x
            mu1_new = mu1
            mu2_new = mu2
        return N1 * np.exp(-(x_new - mu1_new)**2 / variance) \
            + N2 * np.exp(-(x_new - mu2_new)**2 / variance)

    p, c = spo.curve_fit(fit_func, x_axis, data,
                         p0=(variance_guess, mu1_guess, mu2_guess,
                             0.09, 0.01))

    func = fit_func(fit_array, *p)
    plt.plot(fit_array, func, 'k', linewidth=3,
             label="Best Bi-Maxwellian fit")

    # fitting individual gaussians around core and beam peaks
    indexes = peakutils.indexes(func, thres=0.001, min_dist=1)
    peak1 = indexes[0]
    peak2 = indexes[1]

    if E_plot:
        fit_array1 = np.sqrt(fit_array)
    else:
        fit_array1 = fit_array

    parameters1 = peakutils.gaussian_fit(fit_array1[peak1-10:peak1+10],
                                         func[peak1-10:peak1+10],
                                         center_only=False)
    parameters2 = peakutils.gaussian_fit(fit_array1[peak2-10:peak2+10],
                                         func[peak2-10:peak2+10],
                                         center_only=False)
    fit1 = peakutils.gaussian(fit_array1, *parameters1)
    fit2 = peakutils.gaussian(fit_array1, *parameters2)

    def f1(x):
        if E_plot:
            x = np.sqrt(x)
        return peakutils.gaussian(x, *parameters1) \
            - (peakutils.gaussian(parameters1[1], *parameters1))/2.

    def f2(x):
        if E_plot:
            x = np.sqrt(x)
        return peakutils.gaussian(x, *parameters2) \
            - (peakutils.gaussian(parameters2[1], *parameters2))/2.

    x1 = 2000 if E_plot else 600  # estimates of zeros for core and beam
    x2 = 3000 if E_plot else 800
    x3 = 4000 if E_plot else 850
    x4 = 5000 if E_plot else 1000
    zeros1 = spo.root(f1, [x1, x2])
    zeros2 = spo.root(f2, [x3, x4])
    print("1: ", zeros1.x[0], zeros1.x[1])
    print("2: ", zeros2.x[0], zeros2.x[1])
    fwhm1 = zeros1.x[1] - zeros1.x[0]
    fwhm2 = zeros2.x[1] - zeros2.x[0]

    plt.plot(fit_array, fit1, 'r--',
             label="Best Gaussian core fit (FWHM = %g %s)"
             % (fwhm1, "eV" if E_plot else "km/s"))
    plt.plot(fit_array, fit2, 'g--',
             label="Best Gaussian beam fit (FWHM = %g %s)"
             % (fwhm2, "eV" if E_plot else "km/s"))

    return fwhm1, fwhm2


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

    # vz_cut1 = vz_m[vz_m < mu1_guess]
    # vz_cut2 = vz_m[vz_m > mu2_guess]
    # total_cut = total[np.logical_or(v_band_centres < mu1_guess, v_band_centres > mu2_guess)]
    # v_band_centres_cut = v_band_centres[np.logical_or(v_band_centres < mu1_guess, v_band_centres > mu2_guess)]

    # print(v_band_centres.shape)
    # print(total_cut.shape)
    # total_cut = Signal_Count(vz_cut1, True)*1e9 \
        # + Signal_Count(vz_cut2, False)*1e9
