import numpy as np
import quaternion as quat
import timeit


def rotate(vector, B, z_axis, x_axis):
    """Rotating a vector using quaternions
    for rotation by a given angle around a given axis;
    B and z are 3D row vectors;
    Do z rotation first, and then x rotation"""

    B = B / np.linalg.norm(B)
    z = z_axis / np.linalg.norm(z_axis)
    x = x_axis / np.linalg.norm(x_axis)
    zrot_vector = np.cross(B, z)
    xrot_vector = np.cross(B, x)

    if np.linalg.norm(zrot_vector) != 0 and np.linalg.norm(xrot_vector) != 0:
        # Iff B and z neither parallel nor anti-parallel
        vec = np.array([0.] + vector)
        # print(vec)
        zaxis = np.array([0.] + zrot_vector)
        zrot_axis = zaxis / np.linalg.norm(zaxis)
        xaxis = np.array([0.] + xrot_vector)
        xrot_axis = xaxis / np.linalg.norm(xaxis)
        # print("axis:", rot_axis)
        zrot_angle = np.arccos(np.dot(B, z))  # B and z are normalised
        xrot_angle = np.arccos(np.dot(B, x))  # B and x are normalised
        # print("angle: ", rot_angle)
        zaxis_angle = (zrot_angle*0.5) * zrot_axis
        xaxis_angle = (xrot_angle*0.5) * xrot_axis
        v = quat.quaternion(*vec)
        z_exp = quat.quaternion(*zaxis_angle)
        x_exp = quat.quaternion(*xaxis_angle)
        qz = np.exp(z_exp)
        qx = np.exp(x_exp)

        v_rot = (qx * qz * v * np.conjugate(qz) * np.conjugate(qx)).imag

    elif np.dot(B, z) > 0:
        # B and z parallel
        v_rot = vector
    else:
        # B and z anti-parallel
        v_rot = -vector

    if (v_rot-vector).all == 0:
        print(v_rot-vector)

    return v_rot
