from SPC_Plates import *
from SPC_Plot import *
from VDF import *


def integrand_I(vz, vy, vx, v, n):
    """integral returns total current"""
    core = RotMW(vz, vy, vx, v, True, n, B0)
    beam = RotMW(vz, vy, vx, v, False, n, B0)
    # core = BiMax(vz, vy, vx, v, True, n)
    # beam = BiMax(vz, vy, vx, v, False, n)
    vdf = core + beam
    return cst.e * np.sqrt(vz**2 + vy**2 + vx**2) * Area(vz, vy, vx) * vdf


def integrand_V(vz, vy, vx, v_alf, n):
    """integral returns how much more current is in the
    top hemisphere compared to bottom hemisphere"""
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    th = -np.arcsin(vx/v)
    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    angular_range = np.pi / 6
    if -angular_range < th < angular_range:
        return integrand_I(vz, vy, vx, v_alf, n) * H(th)
        # return integrand_I(vz, vy, vx, v_alf, n) * (th / angular_range)
    else:
        return 0


def integrand_W(vz, vy, vx, v_alf, n):
    """integral returns how much more current is in the
    left hemisphere compared to right hemisphere"""

    v = np.sqrt(vx**2 + vy**2 + vz**2)
    th = np.arcsin(vy/v)
    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    angular_range = np.pi / 6
    if -angular_range < th < angular_range:
        # return integrand_I(vz, vy, vx, v_alf, n) * (th / angular_range)
        return integrand_I(vz, vy, vx, v_alf, n) * H(th)
    else:
        return 0


def integrand_plate(vz, vy, vx, v_alf, n, plate):
    """integral returns the current in a given detector plate"""

    v = np.sqrt(vx**2 + vy**2 + vz**2)
    th_x = -np.arcsin(vx/v)
    th_y = np.arcsin(vy/v)

    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    angular_range = np.pi / 6

    if -angular_range < th_x < angular_range \
            and -angular_range < th_y < angular_range:
        return integrand_I(vz, vy, vx, v_alf, n) * Detector(th_x, th_y, plate)
    else:
        return 0
