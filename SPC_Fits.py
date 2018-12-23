import numpy as np
import peakutils
import scipy.optimize as spo
import matplotlib.pyplot as plt


def FWHM(E_plot, is_core, x_axis, data, fit_array, mu_guess, variance_guess):
    """Get FWHM of the data"""

    def fit_func(x, variance, mu, N):
        """1D maxwellian fit"""
        if E_plot:
            x = np.sqrt(x)
            mu = np.sqrt(mu)
        return N * np.exp(-(x - mu)**2 / variance)

    p, c = spo.curve_fit(fit_func, x_axis, data,
                         p0=(variance_guess, mu_guess, 0.09))

    if E_plot:

        def new_fit(x):
            return fit_func(x, *p) - (fit_func(p[1], *p) / 2.)

        x1 = 2000 if is_core else 4000  # estimates of zeros for core and beam
        x2 = 3000 if is_core else 5000
        zeros = spo.root(new_fit, [x1, x2])
        print(zeros.x)
        fwhm = zeros.x[1] - zeros.x[0]

    else:
        # Either follow the same proceedure as above to get fwhm
        # or use the analytical expression below as they are equivalent
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
            x = np.sqrt(x)
            mu1 = np.sqrt(mu1)
            mu2 = np.sqrt(mu2)
        return N1 * np.exp(-(x - mu1)**2 / variance) \
            + N2 * np.exp(-(x - mu2)**2 / variance)

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

    p, c = spo.curve_fit(fit_func, x_axis, data,
                         p0=(variance_guess, mu1_guess, mu2_guess,
                             0.09, 0.01))

    func = fit_func(fit_array, *p)
    plt.plot(fit_array, func, 'k', linewidth=3,
             label="Best Bi-Maxwellian fit")

    # fitting individual gaussians around core and beam peaks
    indexes = peakutils.indexes(func, thres=0.001, min_dist=1)

    if E_plot:
        fit_array1 = np.sqrt(fit_array)
    else:
        fit_array1 = fit_array

    peak1 = indexes[0]

    parameters1 = peakutils.gaussian_fit(fit_array1[peak1-10:peak1+10],
                                         func[peak1-10:peak1+10],
                                         center_only=False)
    fit1 = peakutils.gaussian(fit_array1, *parameters1)

    x1 = 2000 if E_plot else 650  # estimates of zeros for core and beam
    x2 = 3000 if E_plot else 800
    x3 = 4000 if E_plot else 930
    x4 = 5000 if E_plot else 1000
    zeros1 = spo.root(f1, [x1, x2])
    print("1: ", zeros1.x[0], zeros1.x[1])
    fwhm1 = zeros1.x[1] - zeros1.x[0]
    plt.plot(fit_array, fit1, 'r--',
             label="Best Gaussian core fit (FWHM = %g %s)"
             % (fwhm1, "eV" if E_plot else "km/s"))

    if len(indexes) > 1:
        peak2 = indexes[1]
        parameters2 = peakutils.gaussian_fit(fit_array1[peak2-10:peak2+10],
                                             func[peak2-10:peak2+10],
                                             center_only=False)
        fit2 = peakutils.gaussian(fit_array1, *parameters2)

        zeros2 = spo.root(f2, [x3, x4])
        print("2: ", zeros2.x[0], zeros2.x[1])
        fwhm2 = zeros2.x[1] - zeros2.x[0]

        plt.plot(fit_array, fit2, 'g--',
                 label="Best Gaussian beam fit (FWHM = %g %s)"
                 % (fwhm2, "eV" if E_plot else "km/s"))
    else:
        fwhm2 = 0

    return fwhm1, fwhm2
