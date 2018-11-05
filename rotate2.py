import numpy as np
import math


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
            ((w_y*sin_angle + w_x*w_z*(1-cos_angle)) * z)
    rot_y = ((w_z*sin_angle + w_x*w_y*(1-cos_angle)) * x) + \
            ((cos_angle + np.square(w_y)*(1-cos_angle)) * y) + \
            ((-w_x*sin_angle + w_y*w_z*(1-cos_angle)) * z)
    rot_z = ((-w_y*sin_angle + w_x*w_z*(1-cos_angle)) * x) + \
            ((w_x*sin_angle + w_y*w_z*(1-cos_angle)) * y) + \
            ((cos_angle + np.square(w_z)*(1-cos_angle)) * z)
    """

    v = np.array([x, y, z])
    rot_v = v + sin_angle*np.cross(rot_axis, v) + (1 - cos_angle)*np.cross(rot_axis, np.cross(rot_axis, v))
    rot_x, rot_y, rot_z = rot_v

    return rot_x, rot_y, rot_z


def axis_transform(x, y, z, old_axis, new_axis):
    cos_angle = np.dot(old_axis, new_axis) / (np.linalg.norm(old_axis) * np.linalg.norm(new_axis))
    rot_axis = np.cross(old_axis, new_axis)
    rot_axis_length = np.linalg.norm(rot_axis)
    if rot_axis_length != 0:
        rot_axis_norm = rot_axis / rot_axis_length
        rot_angle = -np.arccos(cos_angle)
        print(rot_angle)

        new_x, new_y, new_z = rodrigues_rotation_formula(x, y, z, rot_axis_norm, rot_angle)

    elif cos_angle > 0:
        print('hi')
        new_x, new_y, new_z = x, y, z

    else:
        print('bob')
        new_x, new_y, new_z = -x, -y, -z

    return new_x, new_y, new_z


def coordinate_transform(x, y, z, new_x_axis, new_y_axis, new_z_axis):

    rot_x, _, _ = axis_transform(x, y, z, [1, 0, 0], new_x_axis)

    _, rot_y, _ = axis_transform(x, y, z, [0, 1, 0], new_y_axis)

    _, _, rot_z = axis_transform(x, y, z, [0, 0, 1], new_z_axis)

    print(rot_x, rot_y, rot_z)


if __name__ == '__main__':
    coordinate_transform(1, 1, 1, [-1, 0, 0], [0, -1, 0], [0, 0, -1])
