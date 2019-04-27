import numpy as np
import scipy.constants as cst
import csv
import os


constants = {
    "n": 92e6,  # m^-3
    "B": 108e-9  # T
}

E_plot = False
Plates = True
total = False
Core = True
core_fraction = 0.8 if total else 1

load = True
perturbed = False
comment = "final_"
Rot = "Non-radial"
# Rot = "Big-deflection"
N = 100

pi = np.pi
xthermal_speed = ((cst.k * 2.4e5) / cst.m_p)**0.5
ythermal_speed = ((cst.k * 2.4e5) / cst.m_p)**0.5
# zthermal_speed = ((cst.k * constants["T_par"]) / cst.m_p)**0.5
# radial_temp = zthermal_speed**2 * (cst.m_p / cst.k)
# print("speed, temp: ", zthermal_speed, radial_temp)

B_dict = {
    1: np.array([(-0.2**0.5), (-0.3**0.5), (-0.5**0.5)]),
    2: np.array([(0.1**0.5), (0.6**0.5), (0.3**0.5)]),
    3: np.array([(0.2**0.5), (0.1**0.5), (-0.7**0.5)]),
    'Radial': np.array([0, 0, 1]),
    5: np.array([(0.2**0.5), (0.2**0.5), (0.6**0.5)]),
    'ZX': np.array([(0.5**0.5), 0, (0.5**0.5)]),
    'SPAN -X,-Y': np.array([(-0.5**0.5), 0, (0.5**0.5)]),
    'ZY': np.array([0, (0.5**0.5), (0.5**0.5)]),
    8: np.array([(-0.1**0.5), (0.1**0.5), (0.8**0.5)]),
    'Non-radial': np.array([0.1, 0.2, 1]),
    'Big-deflection': np.array([0.3, 0.5, 0.5])
    }

B_hat = B_dict[Rot] / np.linalg.norm(B_dict[Rot])
# print("B0: ", B_hat, Rot)
B0 = B_hat * constants["B"]
dB = constants["B"] * 1e-3 * np.array([(0.3**0.5), (0.5**0.5), (0.2**0.5)])
# print("dB: ", dB)
B = (B0 + dB) if perturbed else B0
# print("B: ", B)

theta_0 = np.dot(B0, np.array([0, 0, 1])) / np.linalg.norm(B0)
theta_BR = np.dot(B, np.array([0, 0, 1])) / np.linalg.norm(B)
# print("thetas: ", theta_BR, theta_0)


# B-field dependent alfven velocity for protons
va = np.linalg.norm(B0) / np.sqrt(cst.mu_0 * constants["n"] * cst.m_p)
# va = 88000  # m/s
# print("V_alf: ", va)


# v_sw = np.array([20000, 200000, 400000])  # solar wind bulk velocity in m/s
# bulk_speed = np.array([15000, 20000, 700000])  # sw bulk velocity in m/s
bulk_speed = np.array([50000, 50000, 700000])  # sw bulk velocity in m/s
v_sc = np.array([0, 0, 0])  # space craft velocity in m/s
# alfvenic fluctuation
dv = va * (-np.cos(theta_BR) + np.cos(theta_0)) * B/np.linalg.norm(B)
v_sw = (bulk_speed + dv) if perturbed else bulk_speed
# print("speeds: ", bulk_speed, v_sw)
beam_v = (va * B_hat) + v_sw

print(Rot, bulk_speed, N)

# print("V_sw: ", v_sw)
# print("V_beam: ", beam_v)
lim = 2e6  # integration limit for SPC (use instead of np.inf)

"""SPC has an energy measurement range of 100 eV - 8 keV
corresponding to these velocities in m / s"""
J = 1.6e-19  # multiply for ev to J conversions
band_low = np.sqrt((2 * 100 * J) / cst.m_p)  # 138 km / s
band_high = np.sqrt((2 * 8e3 * J) / cst.m_p)  # 1237 km / s
# sigma = np.sqrt((2 * zthermal_speed * J) / cst.m_p) / 1000
# print("sigma: ", sigma, zthermal_speed/1000)



# if load:
    # print("dictionary B: ", mydict['Bx'], mydict['By'], mydict['Bz'])
