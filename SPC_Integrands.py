from SPC_Plates import *
from SPC_Plot import *


def integrand_I(vz, vy, vx, v, n):
    vdf = BiMax(vz, vy, vx, v, True, n) + BiMax(vz, vy, vx, v, False, n)
    return cst.e * np.sqrt(vz**2 + vy**2 + vx**2) * Area(vz, vy, vx) * vdf


def integrand_V(vz, vy, vx, v_alf, n):
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    th = np.arcsin(vy/v)
    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    if -np.pi / 6 < th < np.pi / 6:
        return integrand_I(vz, vy, vx, v_alf, n) * H(th)
    else:
        return 0


def integrand_W(vz, vy, vx, v_alf, n):
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    th = -np.arcsin(vx/v)
    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    if -np.pi / 6 < th < np.pi / 6:
        return integrand_I(vz, vy, vx, v_alf, n) * H(th)
    else:
        return 0


def integrand_plate(vz, vy, vx, v_alf, n, plate):
    """integral returns the current in a given detector plate"""
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    th_x = -np.arcsin(vx/v)
    th_y = np.arcsin(vy/v)
    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    if -np.pi / 6 < th_x < np.pi / 6 and -np.pi / 6 < th_y < np.pi / 6:
        return integrand_I(vz, vy, vx, v_alf, n) * Detector(th_x, th_y, plate)
    else:
        return 0
