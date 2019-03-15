import numpy as np
import Global_Variables as gv
import scipy.constants as cst
from SPC_Fits import *

print(gv.comment)
n1 = 0.03
n2 = 0.01

band_centres = np.genfromtxt("./Data/Band_Centres_%s_%s_%s.csv"
                             % ("Energy" if gv.E_plot else "Velocity",
                                "N_%g" % gv.N, "Field_%s" % gv.Rot))

data1 = np.genfromtxt('%s_quad_1.csv' % filename)
data2 = np.genfromtxt('%s_quad_2.csv' % filename)
data3 = np.genfromtxt('%s_quad_3.csv' % filename)
data4 = np.genfromtxt('%s_quad_4.csv' % filename)
total_quads = data1 + data2 + data3 + data4
fit_array = np.linspace(np.min(band_centres), np.max(band_centres), gv.N)

# core = Total_Fit(gv.E_plot, band_centres, total_quads, fit_array, gv.total,
#                  gv.v_sw[2] / 1e3, gv.beam_v[2] / 1e3, 40,  n1, n2)[1]
#
# if gv.total:
#     beam = Total_Fit(gv.E_plot, band_centres, total_quads, fit_array, gv.total,
#                      gv.v_sw[2] / 1e3, gv.beam_v[2] / 1e3, 40, n1, n2)[1]

vel = np.sqrt((2 * fit_array * gv.J)
              / cst.m_p) if gv.E_plot else fit_array * 1e3

v = np.linalg.norm(gv.v_sw)

area = 1.36e-4  # m^2


d1 = Total_Fit(gv.E_plot, band_centres, data1, fit_array, gv.total,
               700, 900, 40, n1, n2)[0]
d2 = Total_Fit(gv.E_plot, band_centres, data2, fit_array, gv.total,
               700, 900, 40, n1, n2)[0]
d3 = Total_Fit(gv.E_plot, band_centres, data3, fit_array, gv.total,
               700, 900, 40, n1, n2)[0]
d4 = Total_Fit(gv.E_plot, band_centres, data4, fit_array, gv.total,
               700, 900, 40, n1, n2)[0]


n1 = np.sum(d1 / (1e9 * cst.e * vel * area))
n2 = np.sum(d2 / (1e9 * cst.e * vel * area))
n3 = np.sum(d3 / (1e9 * cst.e * vel * area))
n4 = np.sum(d4 / (1e9 * cst.e * vel * area))

print("n: ", n1, n2, n3, n4, n1+n2+n3+n4)
