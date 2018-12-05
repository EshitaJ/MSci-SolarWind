from SPC_Plates_mpmath import *
from SPC_Plot_mpmath import *


def integrand_I(vz, vy, vx, v, n):
    vdf = BiMax(vz, vy, vx, v, True, n) + BiMax(vz, vy, vx, v, False, n)
    return cst.e * mpm.sqrt(vz**2 + vy**2 + vx**2) * Area(vz, vy, vx) * vdf


def integrand_V(vz, vy, vx, v, n):
    v = mpm.sqrt(vx**2 + vy**2 + vz**2)
    th = mpm.asin(vy/v)
    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    if -mpm.pi / 6 < th < mpm.pi / 6:
        return integrand_I(vz, vy, vx, v, n) * H(th)
    else:
        return 0


def integrand_W(vz, vy, vx, v, n):
    v = mpm.sqrt(vx**2 + vy**2 + vz**2)
    th = -mpm.asin(vx/v)
    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    if -mpm.pi / 6 < th < mpm.pi / 6:
        return integrand_I(vz, vy, vx, v, n) * H(th)
    else:
        return 0


def integrand_plate(vz, vy, vx, v_alf, n, plate):
    """integral returns the current in a given detector plate"""
    v = mpm.sqrt(vx**2 + vy**2 + vz**2)
    th_x = -mpm.asin(vx/v)
    th_y = mpm.asin(vy/v)
    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    if -mpm.pi / 6 < th_x < mpm.pi / 6 and -mpm.pi / 6 < th_y < mpm.pi / 6:
        return integrand_I(vz, vy, vx, v_alf, n) * Detector(th_x, th_y, plate)
    else:
        return 0
