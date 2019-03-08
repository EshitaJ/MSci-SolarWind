import numpy as np
import scipy.optimize as spo
from scipy.special import erf
import scipy.constants as cst
import matplotlib.pyplot as plt
import Global_Variables as gv
import SPC_Fits as scfit
import seaborn as sns
sns.set()

data1 = np.genfromtxt(
    "./Data/Test-Run_%s_%s_N_%g_Field_%s_quad_1.csv"
    % (
        "Perturbed" if gv.perturbed else "",
        "total" if gv.total else "core", gv.N, gv.Rot)
    )
data2 = np.genfromtxt(
    "./Data/Test-Run_%s_%s_N_%g_Field_%s_quad_2.csv"
    % (
        "Perturbed" if gv.perturbed else "",
        "total" if gv.total else "core", gv.N, gv.Rot)
    )
data3 = np.genfromtxt(
    "./Data/Test-Run_%s_%s_N_%g_Field_%s_quad_3.csv"
    % (
        "Perturbed" if gv.perturbed else "",
        "total" if gv.total else "core", gv.N, gv.Rot)
    )
data4 = np.genfromtxt(
    "./Data/Test-Run_%s_%s_N_%g_Field_%s_quad_4.csv"
    % (
        "Perturbed" if gv.perturbed else "",
        "total" if gv.total else "core", gv.N, gv.Rot)
    )
xdata = np.genfromtxt("./Data/Band_Centres_%s_%s_%s.csv"
                      % ("Energy" if gv.E_plot else "Velocity",
                         "N_%g" % gv.N, "Field_%s" % gv.Rot),
                      )

total = (data1 + data2 + data3 + data4)
vdata = (data1 + data2 - data3 - data4) / total
wdata = (data1 + data4 - data2 - data3) / total

# peak = np.argmax(data1)
# xdata = xdata[peak-15:peak+50]
# vdata = vdata[peak-15:peak+50]
# wdata = wdata[peak-15:peak+50]
# data4 = data4[peak-15:peak+50]
# total = total[peak-15:peak+50]

k_B = cst.physical_constants["Boltzmann constant in eV/K"][0]
a_guess = 0.01
b_guess = -800

fit_array = np.linspace(np.min(xdata), np.max(xdata), 1e2)
# cdf1 = scfit.Total_Fit(gv.E_plot, xdata, data1, fit_array, gv.total, 700, 900, 44, lbl='1')
# cdf2 = scfit.Total_Fit(gv.E_plot, xdata, data2, fit_array, gv.total, 700, 900, 44, lbl='2')
# cdf3 = scfit.Total_Fit(gv.E_plot, xdata, data3, fit_array, gv.total, 700, 900, 44, lbl='3')
# scfit.Total_Fit(gv.E_plot, xdata, data4, fit_array, gv.total, 700, 900, 50)

# plt.plot(xdata, data1, 'kx', label='Quadrant 1')
# plt.plot(xdata, data2, 'rx', label='Quadrant 2')
# plt.plot(xdata, data3, 'gx', label='Quadrant 3')
# plt.plot(xdata, data4, 'bx', label='Quadrant 4')


def f(x, a, b):
    return erf(a * (x + b))


# xdata1 = np.linspace(np.min(xdata), np.max(xdata), len(cdf1))
# p_v, c_v = spo.curve_fit(f, xdata1, cdf1, p0=(a_guess, b_guess))
# p_w, c_w = spo.curve_fit(f, xdata1, cdf3, p0=(a_guess, b_guess))
p_w, c_w = spo.curve_fit(f, xdata, wdata, p0=(a_guess, b_guess))
p_v, c_v = spo.curve_fit(f, xdata, vdata, p0=(a_guess, b_guess))
# g = f(xdata1, a_guess, b_guess)
#
# # for ga, gb in zip([1e-2, 1e-3, 1e-4], [-700] * 4):
# #     plt.plot(xdata1, f(xdata1, ga, gb), label="$a=%g, b=%g$" % (ga, gb))
#
# #
# #
# print("P_w: ", p_w)
# print("P_v: ", p_v)
# # print("P_4: ", p_4)
#
# # a = p_v[0]
#
# # aw = p_w[0]
#
# # plt.plot(xdata1, g)
# # Ty_reco = cst.m_p / (2 * cst.k * b**2)
# # print("Y Temperature: ", Ty_reco)
# # print("T_y: ", gv.constants["T_y"])
# #
# # Tx_reco = cst.m_p / (2 * cst.k * bw**2)
# # print("X Temperature: ", Tx_reco)
# # print("T_x: ", gv.constants["T_x"])
#
plt.plot(xdata, vdata, 'bx', label="v")
plt.plot(xdata, wdata, 'gx', label="w")
func_w = f(xdata, *p_w)
func_v = f(xdata, *p_v)
# # func_4 = f(xdata*1e3, *p_4)
# # print("func4: ", func_4)
#
# plt.plot(xdata1, func_v, 'k-', label='func v')
# plt.plot(xdata1, func_w, 'b-', label='func w')
plt.legend()
plt.show()
