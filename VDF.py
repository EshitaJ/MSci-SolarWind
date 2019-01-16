import numpy as np
import scipy.constants as cst
from Global_Variables import *
from Quaternion import *


def BiMax(z, y, x, v, is_core, n):
    """3D Bi-Maxwellian distribution;
    Well defined only in the B field frame, with main axis = B_z"""
    T_x = constants["T_x"]  # K
    T_y = constants["T_y"]
    T_z = constants["T_z"]
    core_fraction = 0.9

    y_n = y
    x_n = x

    if is_core:
        n_p = core_fraction * n
        z_new = z - 700000
    else:
        n_p = (1 - core_fraction) * n
        z_new = z - v - 700000

    norm = n_p * (cst.m_p/(2 * np.pi * cst.k))**1.5 / np.sqrt(T_x * T_y * T_z)
    exponent = -((z_new**2/T_z) + (y_n**2/T_y) + (x_n**2/T_x)) \
        * (cst.m_p/(2 * cst.k))
    vdf = norm * np.exp(exponent)
    return vdf


def rotationmatrix(B, z_axis, bool):
    """Rotation matrix given according to Rodriugues' rotation formula
    for rotation by a given angle around a given axis;
    B and z are 3D row vectors;
    Either rotate B(VDF) onto z(SPC) or vice versa"""
    B = B / np.linalg.norm(B)
    z = z_axis / np.linalg.norm(z_axis)
    if bool:
        rot_vector = np.cross(z, B)
    else:
        rot_vector = np.cross(B, z)
    # print(B, z, rot_vector)
    if np.linalg.norm(rot_vector) != 0:
        rot_axis = rot_vector / np.linalg.norm(rot_vector)
        cos_angle = np.dot(z, B)  # B and z are normalised
        # print("Axis, angle: ", rot_axis, np.arccos(cos_angle))
        cross_axis = np.array(([0, -rot_axis[2], rot_axis[1]],
                              [rot_axis[2], 0, -rot_axis[0]],
                              [-rot_axis[1], rot_axis[0], 0]))
        outer_axis = np.outer(rot_axis, rot_axis)
        R = np.identity(3)*cos_angle + cross_axis*np.sqrt(1 - cos_angle**2) \
            + (1 - cos_angle)*outer_axis
    elif np.dot(z, B) > 0:
        # B and z parallel
        R = np.identity(3)
    else:
        # B and z anti-parallel
        R = -np.identity(3)
    # print(R)
    return R


def rotatedMW(vz, vy, vx, v, is_core, n, B):
    T_x = constants["T_x"]  # K
    T_y = constants["T_y"]
    T_z = constants["T_z"]
    core_fraction = 0.9

    vel = np.array([vx, vy, vz])  # in SPC frame
    v_beam = np.array([0, 0, v])

    if is_core:
        n_p = core_fraction * n
        v_new = vel - v_sw
    else:
        n_p = (1 - core_fraction) * n
        v_new = vel - v_sw - v_beam

    R = rotationmatrix(B, np.array([0, 0, 1]), True)
    V_rotated = np.dot(R, v_new)  # - v_sc
    # print("R: ", R)
    # print("V: ", V_rotated-v_new)
    x, y, z = V_rotated
    # x = x1 - 10000

    norm = n_p * (cst.m_p/(2 * np.pi * cst.k))**1.5 / np.sqrt(T_x * T_y * T_z)
    exponent = -((z**2/T_z) + (y**2/T_y) + (x**2/T_x)) * (cst.m_p/(2 * cst.k))
    DF = norm * np.exp(exponent)
    return DF


def RotMW(vz, vy, vx, v, is_core, n, B):
    T_x = constants["T_x"]  # K
    T_y = constants["T_y"]
    T_z = constants["T_z"]
    core_fraction = 0.9

    vel = np.array([vx, vy, vz])  # in SPC frame
    v_beam = np.array([0, 0, v])

    if is_core:
        n_p = core_fraction * n
        v_new = vel - v_sw
    else:
        n_p = (1 - core_fraction) * n
        v_new = vel - v_sw - v_beam
    V = rotate(v_new, B, np.array([0, 0, 1]))
    x, y, z = V

    norm = n_p * (cst.m_p/(2 * np.pi * cst.k))**1.5 / np.sqrt(T_x * T_y * T_z)
    exponent = -((z**2/T_z) + (y**2/T_y) + (x**2/T_x)) * (cst.m_p/(2 * cst.k))
    DF = norm * np.exp(exponent)
    return DF
