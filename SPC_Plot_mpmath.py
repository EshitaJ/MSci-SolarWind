import os
import numpy as npy
import seaborn as sns
import scipy.optimize as spo
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.constants as cst
import mpmath as mpm
sns.set()

constants = {
    "n": 92e6,  # m^-3
    "T_x": 14e5,  # K
    "T_y": 14e5,  # K
    "T_z": 30e5,  # K
    "B": 108e-9  # T
}

lim = 2e6
va = 245531.8


"""SPC has an energy measurement range of 100 eV - 8 keV
corresponding to these velocities in m / s"""
ev = 1.6e-19  # multiply for ev to J conversions
band_low = mpm.sqrt((2 * 100 * ev) / cst.m_p)  # 138 km / s
band_high = mpm.sqrt((2 * 8e3 * ev) / cst.m_p)  # 1237 km / s


def H(theta):
    """Function returning relative area on either side of the SPC;
    theta is angle from normal to SPC"""
    m = 1.51  # Value for SPC from alice's  report
    x = mpm.acos(m * mpm.tan(theta))
    return (1 + (2/mpm.pi)*((mpm.sin(x)*mpm.cos(x)) - x))


def BiMax(z, y, x, v, is_core, n):
    """Just call it from VDF later"""
    T_x = constants["T_x"]  # K
    T_y = constants["T_y"]
    T_z = constants["T_z"]
    core_fraction = 0.9

    if is_core:
        n_p = core_fraction * n
        z_new = z - 700000
    else:
        n_p = (1.0 - core_fraction) * n
        z_new = z - v - 700000

    norm_vdf = ((cst.m_p/(2 * mpm.pi * cst.k))**1.5) / mpm.sqrt(T_x * T_y * T_z)
    exponent = -((z_new**2/T_z) + (y**2/T_y) + (x**2/T_x)) * (cst.m_p/(2 * cst.k))
    vdf = n_p * norm_vdf * mpm.exp(exponent)
    return vdf


def Area(vz, vy, vx):
    """Effective area depends on the elevation_angle
    which is a function of velocity"""
    A_0 = 1.36e-4  # m^2
    v = mpm.sqrt(vx**2 + vy**2 + vz**2)
    elevation_angle = mpm.degrees(mpm.atan(vz / v) - mpm.pi/2)

    # can't have negative area
    A1 = 1 - (1.15 * elevation_angle) / 90
    A2 = 1 - (4.5 * (elevation_angle - 24))/90
    minimum = min(A1, A2)
    area = A_0 * minimum  # m^2
    return max(area, 0)


def current_vdensity(vz, vy, vx, v, is_core, n):
    return cst.e * mpm.sqrt(vz**2 + vy**2 + vx**2) * Area(vz, vy, vx) \
            * BiMax(vz, vy, vx, v, is_core, n)


def Signal_Count(bounds, is_core):
    """Note: use values of integration error as error bars in the plots"""
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
    return npy.array(output)


def Data(velocities, is_core):
    if is_core:
        if os.path.isfile("core_signal_data.csv"):
            signal = npy.genfromtxt("core_signal_data.csv")
        else:
            signal = Signal_Count(velocities, is_core) * 1e9

            npy.savetxt("core_signal_data.csv", signal)
            print("Saved, core")
    else:
        if os.path.isfile("beam_signal_data.csv"):
            signal = npy.genfromtxt("beam_signal_data.csv")
        else:
            signal = Signal_Count(velocities, is_core) * 1e9

            npy.savetxt("beam_signal_data.csv", signal)
            print("Saved, beam")
    return signal


def FWHM(is_core, v_range, data, domain, mu_guess, T_guess):
    """Get FWHM of the data"""

    def fit_func(x, T, mu, N):
        """1D maxwellian fit"""
        return N * mpm.exp(-(x - mu)**2 / T)

    p, c = spo.curve_fit(fit_func, v_range, data, p0=(T_guess, mu_guess, 0.01))

    fwhm = 2 * mpm.sqrt(p[0] * mpm.log(2))

    if is_core:
        plt.plot(domain, fit_func(domain, *p),
                 label="Best Maxwell-Boltzmann core fit (FWHM = %g)" % fwhm)
    else:
        plt.plot(domain, fit_func(domain, *p),
                 label="Best Maxwell-Boltzmann beam fit (FWHM = %g)" % fwhm)

    return fwhm


def Plot(is_core, mu_guess, T_guess, num=70):

    # unequal bin widths as potential is varied
    Potential = npy.linspace(100, 8e3, int(num))
    zz = mpm.sqrt((2 * Potential * ev) / cst.m_p)
    signal = Data(zz, is_core)  # velocities in m/s for calculations

    vz = zz / 1e3  # velocities in km/s for plotting purposes
    band_centres = (vz[:-1] + vz[1:]) / 2.0
    plot_vz = npy.linspace(mpm.min(vz), mpm.max(vz), int(1e3))

    FWHM(is_core, band_centres, signal, plot_vz, mu_guess, T_guess)

    if is_core:
        plt.bar(band_centres, signal, width=mpm.diff(vz),
                label="Measured core at $T_z = %g$" % constants["T_z"])
    else:
        plt.bar(band_centres, signal, width=mpm.diff(vz),
                label="Measured beam at $T_z = %g$" % constants["T_z"])
