import numpy as np
import SPC
import SPC_Fits as spcf
import scipy.constants as cst
import Global_Variables as gv
import matplotlib.pyplot as plt

filename = "./Data/%s%s%s_N_%g_Field_%s" \
    % ("Perturbed_" if gv.perturbed else "",
        "Report_v_",
        "total" if gv.total else "core", gv.N, gv.Rot)


band_centres = np.genfromtxt("./Data/Band_Centres_%s_%s_%s.csv"
                             % ("Energy" if gv.E_plot else "Velocity",
                                "N_%g" % gv.N, "Field_%s" % gv.Rot))
#
#
# t13 = np.array([6.71792368e-19, 6.98232989e-13, 5.75566759e-09, 3.07450792e-06,
# 2.66741762e-04, 6.17063781e-03, 5.14085313e-02, 1.87963624e-01,
# 3.46539000e-01, 3.57048194e-01, 2.22542140e-01, 8.93559781e-02,
#  2.43228834e-02, 4.68139734e-03, 6.59908841e-04, 7.01955945e-05,
#  5.77943758e-06, 3.76483773e-07, 1.97769248e-08, 8.51812286e-10,
#  3.05244280e-11, 9.18225186e-13, 2.37369690e-14, 5.26431312e-16,
#  1.01491893e-17, 1.71516576e-19, 2.55999846e-21, 3.39790347e-23,
#  4.03589889e-25])
#
# t11 = np.array([5.09198920e-52, 1.92777924e-34, 4.77712826e-23, 3.58002601e-15,
#  1.21452402e-09, 8.25392007e-06, 2.86150491e-03, 9.64753733e-02,
#  5.00000595e-01, 5.37188723e-01, 1.43413953e-01, 1.08166184e-02,
#  2.58274425e-04, 2.17249669e-06, 7.09834897e-09, 9.82798600e-12,
#  6.22388957e-15, 1.92740657e-18, 3.09438148e-22, 2.71090640e-26,
#  1.35577052e-30, 4.01691861e-35, 7.36726138e-40, 8.56061012e-45,
#  6.50200592e-50, 3.31166558e-55, 1.15758894e-60, 2.83595247e-66,
#  4.96361340e-72])
#
# t12 = np.array([3.32300243e-27, 2.68434557e-18, 1.64088563e-12, 1.70234697e-08,
#  1.17917926e-05, 1.15312743e-03, 2.51953474e-02, 1.65978760e-01,
#  4.04295386e-01, 4.21371799e-01, 2.09848030e-01, 5.45130707e-02,
#  7.93819793e-03, 6.88310959e-04, 3.74038203e-05, 1.33097948e-06,
#  3.22055022e-08, 5.47539485e-10, 6.73005919e-12, 6.13195509e-14,
#  4.23370703e-16, 2.24962644e-18, 9.47400398e-21, 3.17321441e-23,
#  8.60609159e-26, 1.91388481e-28, 3.53002966e-31, 5.45625764e-34,
#  7.13465734e-37])
#
# # t4 = np.array([5.09198920e-52, 1.92777924e-34, 4.77712826e-23, 3.58002601e-15,
# # 1.21452402e-09, 8.25392007e-06, 2.86150491e-03, 9.64753733e-02, 5.00000595e-01,
# # 5.37188723e-01, 1.43413953e-01, 1.08166184e-02, 2.58274425e-04, 2.17249669e-06,
# # 7.09834897e-09, 9.82798600e-12, 6.22388957e-15, 1.92740657e-18, 3.09438148e-22,
# # 2.71090640e-26, 1.35577052e-30, 4.01691861e-35, 7.36726138e-40, 8.56061012e-45,
# # 6.50200592e-50, 3.31166558e-55, 1.15758894e-60, 2.83595247e-66, 4.96361340e-72])
#
#
fit_array = np.linspace(np.min(band_centres), np.max(band_centres), gv.N)
#
#
# def gauss(x, N, mu, sg):
#     return N * np.exp(-((x - mu) / (np.sqrt(2) * sg))**2)
#
#
# v11 = ((cst.k * 1e5) / cst.m_p)**0.5 / 1e3
# v12 = ((cst.k * 2e5) / cst.m_p)**0.5 / 1e3
# v13 = ((cst.k * 3e5) / cst.m_p)**0.5 / 1e3
# # v4 = ((cst.k * 1e5) / cst.m_p)**0.5 / 1e3
#
# # g11 = gauss(band_centres, np.max(t11), band_centres[np.argmax(t11)], v11)
# # g12 = gauss(band_centres, np.max(t12), band_centres[np.argmax(t12)], v12)
# # g13 = gauss(band_centres, np.max(t13), band_centres[np.argmax(t13)], v13)
# # g4 = gauss(band_centres, np.max(t4), band_centres[np.argmax(t4)], v4)
#
#
# plt.figure(1)
# plt.plot(band_centres, t11, 'o', label="data1")
# spcf.Fit(gv.E_plot, band_centres, t11, fit_array,
#          band_centres[np.argmax(t11)], v11, np.max(t11))
# # plt.plot(band_centres, g1, 'x', label="gauss")
# plt.legend()
# plt.figure(2)
# plt.plot(band_centres, t12, '-', label="data2")
# spcf.Fit(gv.E_plot, band_centres, t12, fit_array,
#          band_centres[np.argmax(t12)], v12, np.max(t12))
# # plt.plot(band_centres, g2, 'x', label="gauss")
# # plt.plot(band_centres, t4, '--', label="data4")
# plt.legend()
# plt.figure(3)
# plt.plot(band_centres, t13, 'x', label="data3")
# spcf.Fit(gv.E_plot, band_centres, t13, fit_array,
#          band_centres[np.argmax(t13)], v13, np.max(t13))
# # plt.plot(band_centres, g3, 'x', label="gauss")
# plt.legend()
# # plt.figure(4)
# # spcf.Fit(gv.E_plot, band_centres, t4, fit_array,
#          # band_centres[np.argmax(t4)], v4, np.max(t4))
# # plt.plot(band_centres, g4, 'x', label="gauss")
# # plt.legend()
#
#
# plt.show()


def f(t_perp_guess, t_par_guess):
    I_tot = SPC.main(t_perp_guess, t_par_guess, "TEST", gv.load)
    np.savetxt("1e5_2e5_I.csv", I_tot)
    print(I_tot)
    v_guess = ((cst.k * t_par_guess) / cst.m_p)**0.5 / 1e3
    params = spcf.Fit(gv.E_plot, band_centres, I_tot, fit_array,
             band_centres[np.argmax(I_tot)], v_guess, np.max(I_tot))
    n_estimate, mu_estimate, sg_estimate = params
    print(t_perp_guess, t_par_guess, params)
    # plt.show()


f(1e5, 2e5)
