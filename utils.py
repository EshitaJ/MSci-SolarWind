import scipy.constants as cst
import numpy as np


def BiMax(x, y, z, v_A, T_x, T_y, T_z, n, core_fraction, is_core, bulk_velocity=700000):
    if is_core:
        n_p = core_fraction * n
        y = y - bulk_velocity
    else:
        n_p = (1 - core_fraction) * n
        y = y - v_A - bulk_velocity

    norm = n_p * (cst.m_p/(2 * np.pi * cst.k))**1.5 / np.sqrt(T_x * T_y * T_z)
    exponent = -((z**2/T_z) + (y**2/T_y) + (x**2/T_x)) \
        * (cst.m_p/(2 * cst.k))
    vdf = norm * np.exp(exponent)
    return vdf


def sph_to_cart(r, theta, phi):
    cos_theta = np.cos(theta)

    """x = r * cos_theta * np.cos(phi)
    y = r * cos_theta * np.sin(phi)
    z = r * np.sin(theta)"""

    x = r * np.sin(theta)
    y = r * cos_theta * np.cos(phi)
    z = r * cos_theta * np.sin(phi)

    return x, y, z


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


n_array = np.array([1, 0, 0])
m_array = np.array([0, 1, 0])

n_rotation_matrix = rotationmatrix(n_array, np.array([1, 0, 0]))
m_rotation_matrix = rotationmatrix(np.transpose(np.matmul(n_rotation_matrix, np.transpose(m_array))), np.array([0, 1, 0]))
nm_rotation_matrix = np.matmul(m_rotation_matrix, n_rotation_matrix)


def RotatedBiMaW(x, y, z, v_A, T_x, T_y, T_z, n, core_fraction, is_core, bulk_velocity):

    vel = np.array([-x, -y, -z])  # in SPC frame, -ve due to look direction; turned on here
    v_beam = np.array([v_A, 0, 0])
    #print(nm_rotation_matrix)
    #print(vel)

    v_rotated = np.matmul(nm_rotation_matrix, vel)
    #print(v_rotated)
    v_rotated -= bulk_velocity
    if is_core:
        n_p = core_fraction * n
        v_new = v_rotated
    else:
        n_p = (1 - core_fraction) * n
        v_new = v_rotated - v_beam
    #print(v_rotated)
    x, y, z = v_new

    norm = n_p * (cst.m_p/(2 * np.pi * cst.k))**1.5 / np.sqrt(T_x * T_y * T_z)
    exponent = -((z**2/T_z) + (y**2/T_y) + (x**2/T_x)) * (cst.m_p/(2 * cst.k))
    DF = norm * np.exp(exponent)
    #print('change places!')
    return DF
