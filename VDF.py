import numpy as np
import scipy.constants as cst
from Global_Variables import *


def rotationmatrix(B, axis):
    """Rotation matrix given according to Rodriugues' rotation formula
    for rotation by a given angle around a given axis;
    B and z are 3D row vectors;
    Either rotate B(VDF) onto z(SPC) or vice versa"""
    B = B / np.linalg.norm(B)
    z = axis / np.linalg.norm(axis)
    rot_vector = np.cross(B, z)
    if np.linalg.norm(rot_vector) != 0:
        rot_axis = rot_vector / np.linalg.norm(rot_vector)
        cos_angle = np.dot(B, z)  # B and z are normalised
        # print("Axis, angle: ", rot_axis, np.arccos(cos_angle))
        cross_axis = np.array(([0, -rot_axis[2], rot_axis[1]],
                              [rot_axis[2], 0, -rot_axis[0]],
                              [-rot_axis[1], rot_axis[0], 0]))
        outer_axis = np.outer(rot_axis, rot_axis)
        R = np.identity(3)*cos_angle + cross_axis*np.sqrt(1 - cos_angle**2) \
            + (1 - cos_angle)*outer_axis
    elif np.dot(B, z) > 0:
        # B and z parallel
        R = np.identity(3)
    else:
        # B and z anti-parallel
        R = -np.identity(3)
    # print(R)
    return R


R = rotationmatrix(B, np.array([0, 0, 1]))
print("R: ", R)


def rotatedMW(vz, vy, vx, v, is_core, n, core_fraction):
    T_x = constants["T_x"]  # K
    T_y = constants["T_y"]
    T_z = constants["T_z"]

    vel = np.array([-vx, -vy, -vz])  # in SPC frame, -ve due to look direction
    v_beam = beam_v

    if is_core:
        n_p = core_fraction * n
        v_new = vel + v_sw
    else:
        n_p = (1 - core_fraction) * n
        v_new = vel + v_beam

    V_rotated = np.dot(R, v_new)  # - v_sc
    x, y, z = V_rotated

    norm = n_p * (cst.m_p/(2 * pi * cst.k))**1.5 / np.sqrt(T_x * T_y * T_z)
    exponent = -((z**2/T_z) + (y**2/T_y) + (x**2/T_x)) * (cst.m_p/(2 * cst.k))
    DF = norm * np.exp(exponent)
    return DF
