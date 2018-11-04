import numpy as np
import math

# ecliptic frame: z= up out of plane, x=radially out from Sun, y=perpendicular to x&z via right hand rule


def euler_rodrigues_intrinsic_rotation(x, y, z, rot_axis='z', rot_angle=np.pi/4):
    assert rot_axis in ['x', 'y', 'z'], "This is not a valid intrinsic rotation axis"

    a = math.cos(rot_angle/2)
    if rot_axis == 'x':
        b = math.sin(rot_angle/2)
        c = 0
        d = 0
    elif rot_axis == 'y':
        b = 0
        c = math.sin(rot_angle/2)
        d = 0
    else:
        b = 0
        c = 0
        d = math.sin(rot_angle/2)

    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    ab, ac, ad, bc, bd, cd = a*b, a*c, a*d, b*c, b*d, c*d

    x_rot, y_rot, z_rot = (aa + bb - cc - dd)*x + 2*(bc - ad)*y + 2*(bd + ac)*z, \
                          2*(bc + ad)*x + (aa + cc - bb -dd)*y + 2*(cd - ab)*z, \
                          2*(bd - ac)*x + 2*(cd + ab) + (aa + dd - bb - cc)*z

    return x_rot, y_rot, z_rot


def velocity_shift(x, y, z, dx, dy, dz):
    x, y, z = x + dx, y + dy, z + dz

    return x, y, z


def frame_transform(x, y, z, direction='B-ecliptic', sw_theta=np.pi/4, sw_phi=0):
    # sw_phi defined as -ve if below ecliptic plane and +ve if above
    if direction == 'B-ecliptic':

        if sw_phi is 0:
            x_rot, y_rot = x * math.cos(sw_theta) - y * math.sin(sw_theta),\
                           y * math.cos(sw_theta) + x * math.sin(sw_theta)

        else:

            x_phi, y_phi, z_phi = euler_rodrigues_intrinsic_rotation(
                x,
                y,
                z,
                rot_axis='y',
                rot_angle=sw_phi)

            x_rot, y_rot, z_rot = euler_rodrigues_intrinsic_rotation(
                x_phi,
                y_phi,
                z_phi,
                rot_axis='z',
                rot_angle=sw_theta)

        print(x_rot, y_rot)

    elif direction == 'ecliptic-B':

        if sw_phi is 0:
            x_rot, y_rot = x * math.cos(-sw_theta) - y * math.sin(-sw_theta),\
                           y * math.cos(-sw_theta) + x * math.sin(-sw_theta)

        else:
            x_theta, y_theta, z_theta = euler_rodrigues_intrinsic_rotation(
                x,
                y,
                z,
                rot_axis='z',
                rot_angle=sw_theta)

            x_rot, y_rot, z_rot = euler_rodrigues_intrinsic_rotation(
                x_theta,
                y_theta,
                z_theta,
                rot_axis='y',
                rot_angle=sw_phi)

        print(x_rot, y_rot)


"""
mesh = [x, y, z]
points = zip(*[i.flat for i in mesh])
"""

if __name__ == '__main__':
    x=np.array([5, 7, 8])
    y=np.array([1, 12, 20])
    frame_transform(x, y, 5)
