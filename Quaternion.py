import numpy as np
import quaternion as quat
import timeit


def rotate(vector, B, z_axis):
    """Rotating a vector using quaternions
    for rotation by a given angle around a given axis;
    B and z are 3D row vectors;
    Either rotate B(VDF) onto z(SPC) or vice versa"""

    B = B / np.linalg.norm(B)
    z = z_axis / np.linalg.norm(z_axis)
    rot_vector = np.cross(B, z)

    if np.linalg.norm(rot_vector) != 0:
        # Iff B and z neither parallel nor anti-parallel
        vec = np.array([0.] + vector)
        # print(vec)
        axis = np.array([0.] + rot_vector)
        rot_axis = axis / np.linalg.norm(axis)
        # print("axis:", rot_axis)
        rot_angle = np.arccos(np.dot(B, z))  # B and z are normalised
        # print("angle: ", rot_angle)
        axis_angle = (rot_angle*0.5) * rot_axis
        v = quat.quaternion(*vec)
        exponent = quat.quaternion(*axis_angle)
        q = np.exp(exponent)

        v_rot = (q * v * np.conjugate(q)).imag

    elif np.dot(B, z) > 0:
        # B and z parallel
        v_rot = vector
    else:
        # B and z anti-parallel
        v_rot = -vector

    return v_rot
