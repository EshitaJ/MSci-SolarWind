import scipy.constants as cst
import numpy as np


def BiMax(x, y, z, v_A, T_par, T_perp, n, core_fraction, is_core, bulk_velocity=700000):
    x, y, z = -x, -y, -z
    if is_core:
        n_p = core_fraction * n
        y = y + bulk_velocity
    else:
        n_p = (1 - core_fraction) * n
        y = y + v_A + bulk_velocity

    norm = n_p * (cst.m_p/(2 * np.pi * cst.k))**1.5 / np.sqrt(T_par * np.square(T_perp))
    exponent = -((z**2/T_perp) + (y**2/T_perp) + (x**2/T_par)) \
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
    z = r * cos_theta * -np.sin(phi)

    return x, y, z


def cart_to_sph(x, y, z):
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    theta = np.arctan(x/np.sqrt(np.square(y) + np.square(z)))
    phi = np.arctan2(-z, y)

    return r, theta, phi


def cart_to_sph_arr(arr):
    x, y, z = arr
    return cart_to_sph(x, y, z)


def rotationmatrix(B, axis, fixed_rot_x=False):
    """Rotation matrix given according to Rodriugues' rotation formula
    for rotation by a given angle around a given axis;
    B and z are 3D row vectors;
    Either rotate B(VDF) onto z(SPC) or vice versa"""
    B = B / np.linalg.norm(B)
    z = axis / np.linalg.norm(axis)
    rot_vector = np.cross(B, z)
    """if fixed_rot_x:
        rot_vector = -np.array([1, 0, 0])"""
    if np.linalg.norm(rot_vector) != 0:
        rot_axis = rot_vector / np.linalg.norm(rot_vector)
        cos_angle = np.dot(B, z)  # B and z are normalised
        print("Axis, angle: ", rot_axis, np.arccos(cos_angle))
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


n_array = np.array([0, 0.35, 0.94])
m_array = np.array([-0.39, -0.86, 0.32])
#n_array = np.array([0, 1, 0])
#m_array = np.array([0, 0, -1])

#field_array = np.array([1, 1, 0])
field_array = n_array
field_array = field_array / np.linalg.norm(field_array)

"""n_rotation_matrix = rotationmatrix(np.array([1, 0, 0]), n_array)
m_rotation_matrix = rotationmatrix(np.array([0, 1, 0]), np.matmul(n_rotation_matrix, m_array), fixed_rot_x=True)
nm_rotation_matrix = np.matmul(m_rotation_matrix, n_rotation_matrix)"""
nm_rotation_matrix = np.array([n_array/np.linalg.norm(n_array),
                               m_array/np.linalg.norm(m_array),
                               np.cross(n_array, m_array)/np.linalg.norm(np.cross(n_array, m_array))])
inv_nm_matrix = np.linalg.inv(nm_rotation_matrix)

field_rotation_matrix = rotationmatrix(np.array([1, 0, 0]), -field_array)
inv_field_matrix = np.linalg.inv(field_rotation_matrix)

nm_field_array = np.matmul(nm_rotation_matrix, field_array)


def RotatedBiMaW(x, y, z, v_A, T_par, T_perp, n, core_fraction, is_core, bulk_velocity, sc_velocity=np.array([0, 0, 0])):
    #print(np.array([x, y, z]))
    vel = np.array([x, y, z])  # in SPC frame, -ve due to look direction; turned on here
    v_beam = np.array([-v_A, 0, 0])
    vel = np.matmul(inv_nm_matrix, vel)
    #print(nm_field_array)
    #print(nm_field_array)
    #print(field_rotation_matrix)
    #print(inv_field_matrix)
    #print('change')
    # being rotated wrong!!! Currently done around origin, needs to be done around centre of itself! Do rot after bulk & sc-vel shift?

    #print(np.matmul(n_rotation_matrix, m_array))
    #print("n: ", n_rotation_matrix)
    #print("m:", m_rotation_matrix)
    #print("nm:", nm_rotation_matrix)
    #print("inv:", inv_nm_matrix)
    #print(vel)
    vel += bulk_velocity
    vel += sc_velocity
    vel = np.matmul(inv_field_matrix, vel)
    if is_core:
        n_p = core_fraction * n
        v_new = vel
    else:
        n_p = (1 - core_fraction) * n
        v_new = vel + v_beam
    #print(v_new)
    #print(v_rotated)
    x, y, z = v_new
    #print('Change Places!')

    norm = n_p * (cst.m_p/(2 * np.pi * cst.k ))**1.5 / np.sqrt(T_par * np.square(T_perp))
    exponent = -((z**2/T_perp) + (y**2/T_perp) + (x**2/T_par)) * (cst.m_p/(2 * cst.k))
    DF = norm * np.exp(exponent)
    return DF


"""def rotatedMW(vz, vy, vx, v, is_core, n):
    T_x = constants["T_x"]  # K
    T_y = constants["T_y"]
    T_z = constants["T_z"]
    core_fraction = 0.9

    vel = np.array([-vx, -vy, -vz])  # in SPC frame, -ve due to look direction
    v_beam = np.array([0, 0, v])

    if is_core:
        n_p = core_fraction * n
        v_new = vel + v_sw
    else:
        n_p = (1 - core_fraction) * n
        v_new = vel + v_sw + v_beam

    v_new -= v_sw

    V_rotated = np.matmul(R, v_new)  # - v_sc
    V_rotated += v_sw
    x, y, z = V_rotated
    x -= x1
    y -= y1

    norm = n_p * (cst.m_p/(2 * pi * cst.k))**1.5 / np.sqrt(T_x * T_y * T_z)
    exponent = -((z**2/T_z) + (y**2/T_y) + (x**2/T_x)) * (cst.m_p/(2 * cst.k))
    DF = norm * np.exp(exponent)
    return DF"""