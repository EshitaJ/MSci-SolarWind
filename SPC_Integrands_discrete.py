from SPC_Plates import *
from SPC_Plot import *


def integrand_I(vz, vy, vx, v, n, new_x, new_y, new_z):
    vel = np.array([vx, vy, vz])
    Vx = np.dot(vel, new_x)
    Vy = np.dot(vel, new_y)
    Vz = np.dot(vel, new_z)
    vdf = BiMax(Vz, Vy, Vx, v, True, n) + BiMax(Vz, Vy, Vx, v, False, n)
    return cst.e * np.sqrt(Vz**2 + Vy**2 + Vx**2) * Area(Vz, Vy, Vx) * vdf


def integrand_V(vz, vy, vx, v_alf, n, new_x, new_y, new_z):
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    th = np.arcsin(vy/v)
    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    if -np.pi / 6 < th < np.pi / 6:
        return integrand_I(vz, vy, vx, v_alf, n, new_x, new_y, new_z) * H(th)
    else:
        return 0


def integrand_W(vz, vy, vx, v_alf, n, new_x, new_y, new_z):
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    th = -np.arcsin(vx/v)
    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    if -np.pi / 6 < th < np.pi / 6:
        return integrand_I(vz, vy, vx, v_alf, n, new_x, new_y, new_z) * H(th)
    else:
        return 0


def integrand_plate(vz, vy, vx, v_alf, n, plate, new_x, new_y, new_z):
    """integral returns the current in a given detector plate"""
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    th_x = -np.arcsin(vx/v)
    th_y = np.arcsin(vy/v)
    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    if -np.pi / 6 < th_x < np.pi / 6 and -np.pi / 6 < th_y < np.pi / 6:
        return integrand_I(vz, vy, vx, v_alf, n, new_x, new_y, new_z) * Detector(th_x, th_y, plate)
    else:
        return 0
