import numpy as np
import quaternion as quat
import timeit


def rotate(vector, axis, rot_angle):
    """
    Rotating a vector using quaternions
    for rotation by a given angle around a given axis;
    B and z are 3D row vectors;
    Either rotate B(VDF) onto z(SPC) or vice versa
    """

    if np.linalg.norm(axis) != 0:
        # Iff B and z neither parallel nor anti-parallel
        vec = np.array([0.] + vector)
        # print(vec)
        axis = np.array([0.] + raxis)
        rot_axis = axis / np.linalg.norm(axis)
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
