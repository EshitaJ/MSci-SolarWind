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
pi = np.pi
# B0 = np.array([(0.2**0.5)*B, (0.3**0.5)*B, (0.5**0.5)*B])  # Theta 1
# B0 = np.array([(0.1**0.5)*B, (0.6**0.5)*B, (0.3**0.5)*B])  # Theta 2
# B0 = np.array([(0.2**0.5)*B, (0.1**0.5)*B, (0.7**0.5)*B])  # Theta 3
# B0 = np.array([0, 0, B])  # B field in SC frame  # Theta 4
B0 = np.array([(0.2**0.5)*B, (0.2**0.5)*B, (0.6**0.5)*B])  # Theta 5
# B0 = np.array([(0.5**0.5)*B, (0)*B, (0.5**0.5)*B])  # Theta 6
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

load = False
total = False
Th = 5.1
N = 50
