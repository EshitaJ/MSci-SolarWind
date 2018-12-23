import numpy as np
import quaternion as quat
import timeit

# v = [3, 5, 0]
# axis = [1/np.sqrt(2), -1/np.sqrt(2), 0]
# theta = 0.9553166181245092  # radians, anti-clockwise is positive
#
# vector = np.array([0.] + v)
# rot_axis = np.array([0.] + axis)
# axis_angle = (theta*0.5) * rot_axis/np.linalg.norm(rot_axis)
#
# vec = quat.quaternion(*v)
# qlog = quat.quaternion(*axis_angle)
# q = np.exp(qlog)
#
# v_prime = q * vec * np.conjugate(q)
#
#
# print(v, v_prime.imag)


def rotate(vector, B, z_axis):
    """Rotation matrix given according to Rodriugues' rotation formula
    for rotation by a given angle around a given axis;
    B and z are 3D row vectors;
    Either rotate B(VDF) onto z(SPC) or vice versa"""
    B = B / np.linalg.norm(B)
    z = z_axis / np.linalg.norm(z_axis)
    rot_vector = np.cross(B, z)

    if np.linalg.norm(rot_vector) != 0:

        vec = np.array([0.] + vector)
        print(vec)
        axis = np.array([0.] + rot_vector)
        rot_axis = axis / np.linalg.norm(axis)
        print("axis:", rot_axis)
        rot_angle = np.arccos(np.dot(B, z))  # B and z are normalised
        print("angle: ", rot_angle)
        axis_angle = (rot_angle*0.5) * rot_axis

        v = quat.quaternion(*vec)
        exponent = quat.quaternion(*axis_angle)
        q = np.exp(exponent)

        v_rot = (q * v * np.conjugate(q)).imag

    elif np.dot(B, z) > 0:
        # IS THIS EVEN VALID???
        # B and z parallel
        v_rot = np.array(vector)
    else:
        # B and z anti-parallel
        v_rot = -np.array(vector)

    return v_rot
