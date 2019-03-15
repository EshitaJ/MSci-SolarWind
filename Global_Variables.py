import numpy as np
import scipy.constants as cst
import csv
import argparse
import os

# if "nargparse" not in os.environ:
parser = argparse.ArgumentParser()
parser.add_argument("perp", help="Perpendicular SW temperature, in Kelvin",
                    type=float)
parser.add_argument("par", help="Parallel SW temperature, in Kelvin",
                    type=float)
parser.add_argument("comment", help="Comment or notes about the specific run",
                    type=str)
parser.add_argument("load", help="Loads data if True, creates if False",
                    type=bool)
args = parser.parse_args()

constants = {
    "n": 92e6,  # m^-3
    "T_perp": args.perp,  # K
    "T_par": args.par,  # K
    "B": 108e-9  # T
}

E_plot = False
Plates = True
total = False
Core = True
core_fraction = 0.8 if total else 1

load = args.load
perturbed = False
comment = args.comment
Rot = 'Non-radial'
N = 60

print("t_perp: ", constants["T_perp"])
print("t_par: ", constants["T_par"])
print("load: ", load)

pi = np.pi
xthermal_speed = ((cst.k * constants["T_perp"]) / cst.m_p)**0.5
ythermal_speed = ((cst.k * constants["T_perp"]) / cst.m_p)**0.5
zthermal_speed = ((cst.k * constants["T_par"]) / cst.m_p)**0.5
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
bulk_speed = np.array([15000, 20000, 700000])  # sw bulk velocity in m/s
v_sc = np.array([0, 0, 0])  # space craft velocity in m/s
# alfvenic fluctuation
dv = va * (-np.cos(theta_BR) + np.cos(theta_0)) * B/np.linalg.norm(B)
v_sw = (bulk_speed + dv) if perturbed else bulk_speed
# print("speeds: ", bulk_speed, v_sw)
beam_v = (va * B_hat) + v_sw

# print("V_sw: ", v_sw)
# print("V_beam: ", beam_v)
lim = 2e6  # integration limit for SPC (use instead of np.inf)

"""SPC has an energy measurement range of 100 eV - 8 keV
corresponding to these velocities in m / s"""
J = 1.6e-19  # multiply for ev to J conversions
band_low = np.sqrt((2 * 100 * J) / cst.m_p)  # 138 km / s
band_high = np.sqrt((2 * 8e3 * J) / cst.m_p)  # 1237 km / s
sigma = np.sqrt((2 * zthermal_speed * J) / cst.m_p) / 1000
# print("sigma: ", sigma, zthermal_speed/1000)


par_dict = {
         'n': "%.2E" % constants['n'],
         'T_perp': "%.1E" % constants['T_perp'],
         'T_par': "%.1E" % constants['T_par'],
         'B strength': constants["B"],
         'Bx': B[0],
         'By': B[1],
         'Bz': B[2],
         'Bxz': round(np.degrees(np.arctan(B[0]/B[2])), 2),
         'Byz': round(np.degrees(np.arctan(B[1]/B[2])), 2),
         'Bulkx': round(v_sw[0]/1e3, 2),
         'Bulky': round(v_sw[1]/1e3, 2),
         'Bulkz': round(v_sw[2]/1e3, 2),
         'Beamx': round(beam_v[0]/1e3, 2),
         'Beamy': round(beam_v[1]/1e3, 2),
         'Beamz': round(beam_v[2]/1e3, 2),
         'SCx': v_sc[0]/1e3,
         'SCy': v_sc[1]/1e3,
         'SCz': v_sc[2]/1e3,
         'Alfven speed': va/1e3,
         'N': N,
         'Rot': Rot,
         'Perturbed': perturbed,
         'dB_x': dB[0],
         'dB_y': dB[1],
         'dB_z': dB[2],
         'dv_x': dv[0],
         'dv_y': dv[1],
         'dv_z': dv[2],
         'Total': total,
         'Core': Core,
         'Core fraction': core_fraction
         }


def Param_write(filename):
    output = filename + '_parameters'
    with open(output, 'w') as f:
        for key in par_dict.keys():
            f.write("%s,%s\n" % (key, par_dict[key]))
    print("Saved, parameters")


def Param_read(filename):
    input = filename + '_parameters'
    with open(input, mode='r') as infile:
        reader = csv.reader(infile)
        with open('params_new.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            mydict = {rows[0]: rows[1] for rows in reader}
    # print(mydict)
    return mydict


filename = "./Data/%s%s%s_N_%g_Field_%s" \
            % ("Perturbed_" if perturbed else "",
               comment,
               "total" if total else "core", N, Rot)
mydict = Param_read(filename) if load else Param_write(filename)
# if load:
    # print("dictionary B: ", mydict['Bx'], mydict['By'], mydict['Bz'])
