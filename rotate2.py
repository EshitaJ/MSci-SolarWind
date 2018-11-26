import numpy as np
import math

"""
@TODO
Change to simple dot product method
"""


def find_rot_axis(axis1, axis2):
    rot_axis = np.cross(axis1, axis2)
    print(np.linalg.norm(rot_axis))
    rot_axis_norm = rot_axis / np.linalg.norm(rot_axis)

    return rot_axis_norm


def find_rot_angle(axis1, axis2):
    cos_angle = np.dot(axis1, axis2) / (np.linalg.norm(axis1) * np.linalg.norm(axis2))
    angle = np.arccos(cos_angle)

    return angle


def rodrigues_rotation_formula(x, y, z, rot_axis, rot_angle):
    cos_angle = math.cos(rot_angle)
    sin_angle = math.sin(rot_angle)
    """
    w_x, w_y, w_z = rot_axis

    rot_x = ((cos_angle + np.square(w_x)*(1-cos_angle)) * x) + \
            ((w_x*w_y*(1-cos_angle) - w_z*sin_angle) * y) + \
            ((w_y*sin_angle + w_x*w_z*(1-cos_angle)) * z);
    rot_y = ((w_z*sin_angle + w_x*w_y*(1-cos_angle)) * x) + \
            ((cos_angle + np.square(w_y)*(1-cos_angle)) * y) + \
            ((-w_x*sin_angle + w_y*w_z*(1-cos_angle)) * z)
    rot_z = ((-w_y*sin_angle + w_x*w_z*(1-cos_angle)) * x) + \
            ((w_x*sin_angle + w_y*w_z*(1-cos_angle)) * y) + \
            ((cos_angle + np.square(w_z)*(1-cos_angle)) * z)
    """

    v = np.array([x, y, z])
    rot_axis = np.array(rot_axis)
    rot_v = v*cos_angle + sin_angle*np.cross(rot_axis, v) + (1 - cos_angle)*rot_axis*np.dot(rot_axis, v)
    rot_x, rot_y, rot_z = rot_v

    return rot_x, rot_y, rot_z


def axis_transform(x, y, z, old_axis, new_axis):
    cos_angle = np.dot(old_axis, new_axis) / (np.linalg.norm(old_axis) * np.linalg.norm(new_axis))
    rot_axis = np.cross(old_axis, new_axis)
    rot_axis_length = np.linalg.norm(rot_axis)
    if rot_axis_length != 0:
        rot_axis_norm = rot_axis / rot_axis_length
        rot_angle = np.arccos(cos_angle)

        new_x, new_y, new_z = rodrigues_rotation_formula(x, y, z, rot_axis_norm, rot_angle)

    elif cos_angle > 0:
        new_x, new_y, new_z = x, y, z

    else:
        new_x, new_y, new_z = -x, -y, -z

    #print('hi')
    #print(new_x, new_y, new_z)

    return new_x, new_y, new_z


def b_ecliptic_coordinate_transform(phi, theta, psi=0, direction='B-ecliptic'):
    # uses x convention Euler angles

    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    if direction == 'B-ecliptic':
        """x2, y2, z2 = rodrigues_rotation_formula(x, y, z, [0, 0, 1], psi)
        x3, y3, z3 = rodrigues_rotation_formula(x2, y2, z2, [1, 0, 0], theta)
        x4, y4, z4 = rodrigues_rotation_formula(x3, y3, z3, [0, 0, 1], phi)

        print(x4, y4, z4)
        return x4, y4, z4
        """

        x_ax_2 = rodrigues_rotation_formula(x_axis[0], x_axis[1], x_axis[2], z_axis, psi)
        y_ax_2 = rodrigues_rotation_formula(y_axis[0], y_axis[1], y_axis[2], z_axis, psi)

        y_ax_3 = rodrigues_rotation_formula(y_ax_2[0], y_ax_2[1], y_ax_2[2], x_ax_2, theta)
        z_ax_3 = rodrigues_rotation_formula(z_axis[0], z_axis[1], z_axis[2], x_ax_2, theta)

        x_ax_4 = rodrigues_rotation_formula(x_ax_2[0], x_ax_2[1], x_ax_2[2], z_ax_3, phi)
        y_ax_4 = rodrigues_rotation_formula(y_ax_3[0], y_ax_3[1], y_ax_3[2], z_ax_3, phi)

        return x_ax_4, y_ax_4, z_ax_3

    if direction == 'ecliptic-B':
        pass
        # phi, theta, psi = -phi, -theta, -psi





"""
    rot_x, _, _ = axis_transform(x, y, z, [1, 0, 0], new_x_axis)
    _, rot_y, _ = axis_transform(x, y, z, [0, 1, 0], new_y_axis)
    _, _, rot_z = axis_transform(x, y, z, [0, 0, 1], new_z_axis)
     print(rot_x, rot_y, rot_z)
"""

if __name__ == '__main__':
    value =np.array([2, 3, 7])
    newx, newy, newz = b_ecliptic_coordinate_transform(phi=0.765, theta=0.973, psi=2.71)
    print(np.array([np.dot(value, newx), np.dot(value, newy), np.dot(value, newz)]))
