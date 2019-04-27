import numpy as np
import Global_Variables as gv
import scipy.constants as cst
import scipy.integrate as spi
from SPC_Fits import *
import SPC

total_quads, band_centres = SPC.main(2.4e5, 1.7e5, "n_test_", False)

fit_array = np.linspace(0, 1500, 1e3)

# core = Total_Fit(gv.E_plot, band_centres, total_quads, fit_array, gv.total,
#                  gv.v_sw[2] / 1e3, gv.beam_v[2] / 1e3, 40,  n1, n2)[1]
#
# if gv.total:
#     beam = Total_Fit(gv.E_plot, band_centres, total_quads, fit_array, gv.total,
#                      gv.v_sw[2] / 1e3, gv.beam_v[2] / 1e3, 40, n1, n2)[1]

# vel = fit_array * 1e3

# v = np.linalg.norm(gv.v_sw)

area = 1.36e-4  # m^2

# variance_guess = 40

d1 = Fit(gv.E_plot, band_centres, total_quads, fit_array,
         band_centres[np.argmax(total_quads)],
         40, np.max(total_quads))[2]
# d2 = Fit(gv.E_plot, band_centres, data2, fit_array,
#          band_centres[np.argmax(data2)],
#          variance_guess, np.max(data2))
# d3 = Fit(gv.E_plot, band_centres, data3, fit_array,
#          band_centres[np.argmax(data3)],
#          variance_guess, np.max(data3))
# d4 = Fit(gv.E_plot, band_centres, data4, fit_array,
#          band_centres[np.argmax(data4)],
#          variance_guess, np.max(data4))

n_new = d1[0] / (700000 * 1e9 * cst.e * area) * (gv.N / 1.5)

den = total_quads / (band_centres * 1e3)
nden1 = np.sum(den) / (1e9 * cst.e * area)
n3 = np.sum(np.gradient(total_quads, band_centres*1e3)) / (1e9 * cst.e * area)
print("nden: ", nden1 / 1e7, n3 / 1e7, n_new / 1e7)

# # vel = band_centres[np.argmax(total_quads)] * 1e3
# n1 = np.sum(d1 / (1e9 * cst.e * vel * area))
# n2 = np.sum(d2 / (1e9 * cst.e * vel * area))
# n3 = np.sum(d3 / (1e9 * cst.e * vel * area))
# n4 = np.sum(d4 / (1e9 * cst.e * vel * area))
# n = np.sum((d1+d2+d3+d4) / (1e9 * cst.e * vel * area))
# print("n: ", (n1+n2+n3+n4)/1e7, n/1e7)
