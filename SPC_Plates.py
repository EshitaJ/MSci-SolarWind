import numpy as np
import scipy.constants as cst


def Detector(theta_x, theta_y, plate):
    """Returns area of beam hitting the detector plate
    1: 'first quadrant'
    2: 'second quadrant'
    3: 'third quadrant'
    4: 'fourth quadrant'
    angles provided must be in radians
    (Refer to file for derivations)"""
    m = 1.51  # ratio of aperture radius to aperture vertical displacement
    x = np.arccos(m * np.tan(theta_x))
    y = np.arccos(m * np.tan(theta_y))

    if plate == 1:
        return (3*np.pi/2 - x - y + (np.tan(x) + 2/np.tan(y))*(np.cos(x))**2
                + 3*np.sin(2*y)/2 - 2*np.cos(x)*np.cos(y)) / (2*np.pi)
    elif plate == 2:
        return (x + (np.pi/2) - y - (np.tan(x) + 2/np.tan(y))*(np.cos(x))**2
                - 0.5*np.sin(2*y) + 2*np.cos(x)*np.cos(y)) / (2*np.pi)
    elif plate == 3:
        return (x - (np.pi/2) + y - (np.tan(x) - 2/np.tan(y))*(np.cos(x))**2
                + 0.5*np.sin(2*y) - 2*np.cos(x)*np.cos(y)) / (2*np.pi)
    elif plate == 4:
        return (y + (np.pi/2) - x - (np.tan(y) + 2/np.tan(x))*(np.cos(y))**2
                - 0.5*np.sin(2*x) + 2*np.cos(y)*np.cos(x)) / (2*np.pi)


def H(theta):
    """Function returning relative area on either side of the SPC;
    theta is angle from normal to SPC"""
    m = 1.51  # Value specific to SPC from alice's report
    x = np.arccos(m * np.tan(theta))
    return (1 + (2/np.pi)*((np.sin(x)*np.cos(x)) - x))


def Area(vz, vy, vx):
    """Effective area depends on the elevation_angle
    which is a function of velocity"""
    A_0 = 1.36e-4  # m^2
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    # elevation_angle = np.degrees(np.arctan(vz / v) - np.pi/2)
    elevation_angle = np.degrees(np.pi/2 - (vz/v))

    # can't have negative area
    A1 = 1 - (1.15 * elevation_angle) / 90
    A2 = 1 - (4.5 * (elevation_angle - 24))/90
    minimum = min(A1, A2)
    area = A_0 * minimum  # m^2
    return max(area, 0)
