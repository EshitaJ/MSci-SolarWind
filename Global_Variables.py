import numpy as np
import scipy.constants as cst

constants = {
    "n": 92e6,  # m^-3
    "T_x": 14e5,  # K
    "T_y": 14e5,  # K
    "T_z": 3e5,  # K
    "B": 108e-9  # T
}

B = constants["B"]
B0 = np.array([0.7*B, (2**0.5/10)*B, 0.7*B])  # B field in SC frame
# B0 = np.array([0, B/(2**0.5), B/(2**0.5)])  # B field in SC frame
print("B: ", B0)

v_sw = np.array([0, 0, 700000])  # solar wind velocity in m/s
v_sc = np.array([0, 0, 20000])  # space craft velocity in m/s
# B-field dependent alfven velocity for protons
va = np.linalg.norm(B0) / np.sqrt(cst.mu_0 * constants["n"] * cst.m_p)
print("V_alf: ", va)

lim = 2e6  # integration limit for SPC (use instead of np.inf)

"""SPC has an energy measurement range of 100 eV - 8 keV
corresponding to these velocities in m / s"""
J = 1.6e-19  # multiply for ev to J conversions
band_low = np.sqrt((2 * 100 * J) / cst.m_p)  # 138 km / s
band_high = np.sqrt((2 * 8e3 * J) / cst.m_p)  # 1237 km / s
