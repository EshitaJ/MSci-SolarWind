import numpy as np
import peakutils
import scipy.optimize as spo
import scipy.integrate as spi
import matplotlib.pyplot as plt
from Global_Variables import *
from scipy.signal import find_peaks
import seaborn as sns
sns.set()


def Fit(E_plot, x_axis, data, fit_array, mu_guess, sg_guess, n_guess):
    """Get FWHM of the data"""

    def gauss(x, N, mu, sg):
        return N * np.exp(-((x - mu) / (np.sqrt(2) * sg))**2)

    p, c = spo.curve_fit(gauss, x_axis, data,
                         p0=(n_guess, mu_guess,
                             sg_guess))

    func = gauss(fit_array, *p)
    sigma = p[2]

    plt.plot(fit_array, func, '--',
             label="Gaussian fit"
             "\n $v_{th}$ = %.01f %s"
             % (sigma, "eV" if E_plot else "km/s"))
    return func, sigma


def Total_Fit(E_plot, x_axis, data, fit_array, is_total,
              mu1_guess, mu2_guess, var_guess, N1_guess, N2_guess):
    """Get a fit of total data"""

    def fit_func(x, sg, mu1, mu2, N1, N2):
        """Double Gaussian (bimodal) fit"""
        if E_plot:
            x = np.sqrt(x)
            mu1 = np.sqrt(mu1)
            mu2 = np.sqrt(mu2)
            sg = np.sqrt(sg)
        return N1 * np.exp(-((x - mu1) / (np.sqrt(2) * sg))**2) \
            + N2 * np.exp(-((x - mu2) / (np.sqrt(2) * sg))**2)

    def gauss(x, N, mu, sg):
        return N * np.exp(-((x - mu) / (np.sqrt(2) * sg))**2)

    p, c = spo.curve_fit(fit_func, x_axis, data,
                         p0=(var_guess, mu1_guess,
                             mu2_guess if total else mu1_guess,
                             N1_guess, N2_guess))

    func = fit_func(fit_array, *p)
    sigma = p[0]
    # print("sigma: ", sigma, p[0])
    radial_temp = (sigma*1e3)**2 * (cst.m_p / cst.k)


    def cdf(x):
        cdf_integral = spi.quad(fit_func, -x, x,
                                args=(p[0], p[1], p[2], p[3], p[4]))
        return cdf_integral[0]

    if is_total:
        total_integral = spi.quad(fit_func, min(fit_array), max(fit_array),
                                  args=(p[0], p[1], p[2], p[3], p[4]))
        # print("total_integral: ", total_integral[0])
        # cdf = np.vectorize(cdf)
        # func_cdf = cdf(fit_array) / total_integral[0]
        # plt.plot(fit_array, func_cdf, 'x', label=lbl)

    plt.plot(fit_array, func, 'k-', linewidth=2,
             label="Double Gaussian fit"
             "\n $v_{th}$ = %.01f %s"
             % (sigma, "eV" if E_plot else "km/s"))

    # fitting individual gaussians around core and beam peaks
    indexes = peakutils.indexes(func, thres=0.0001, min_dist=0.0001)
    # indexes, _ = find_peaks(func, height=0.01, distance=1)
    # print(indexes)

    if E_plot:
        fit_array1 = np.sqrt(fit_array)
    else:
        fit_array1 = fit_array

    peak1 = indexes[0]
    # print("peak1: ", fit_array1[peak1])

    parameters1, c1 = spo.curve_fit(gauss, fit_array1[peak1-5:peak1+5],
                                    func[peak1-5:peak1+5],
                                    p0=(0.2, fit_array1[peak1], p[0]))
    fit1 = gauss(fit_array1, *parameters1)
    sigma1 = parameters1[2]
    # print("parameters1: ", parameters1)
    if is_total:
        core_integral = spi.quad(gauss, np.min(fit_array1), np.max(fit_array1),
                                 args=(parameters1[0], parameters1[1], parameters1[2]))
        core_frac = core_integral[0]/total_integral[0]
        # print("core_integral: ", core_integral[0], total_integral[0])

    plt.plot(fit_array, fit1, 'r--',
             label="Gaussian core fit"
             "\n $v_{th}$ = %.01f %s"
             "%s"
             % (sigma1, "eV" if E_plot else "km/s",
             "\n Fraction: %.04f" % core_frac if is_total else ""))

    if len(indexes) > 1:
        peak2 = indexes[1]
        # print("peak2: ", fit_array1[peak2])

        parameters2, c2 = spo.curve_fit(gauss, fit_array1[peak2-5:peak2+5],
                                        func[peak2-5:peak2+5],
                                        p0=(0.2, fit_array1[peak2], p[0]))
        fit2 = gauss(fit_array1, *parameters2)
        sigma2 = parameters2[2]
        # print("parameters2: ", parameters2)
        if is_total:
            beam_integral = spi.quad(gauss, np.min(fit_array1), np.max(fit_array1),
                                    args=(parameters2[0], parameters2[1], parameters2[2]))
            beam_frac = beam_integral[0] / total_integral[0]
            # print("beam_integral: ", beam_integral[0], total_integral[0])

        plt.plot(fit_array, fit2, 'g--',
                 label="Gaussian beam fit"
                 "\n $v_{th}$ = %.01f %s"
                 "%s"
                 % (sigma2, "eV" if E_plot else "km/s",
                 "\n Fraction: %.04f" % beam_frac if is_total else ""))
    else:
        parameters2 = np.zeros(3)
        sigma2 = 0
        fit2 = 0
    # plt.legend()
    # plt.show()

    return p[1], parameters1[1], parameters2[1], sigma, sigma1, sigma2, \
        radial_temp
