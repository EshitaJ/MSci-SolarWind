import numpy as np
import scipy.constants as cst


def Detector(theta_x, theta_y, plate):
    """Returns area of beam hitting the detector plate
    1: 'first quadrant'
    2: 'second quadrant'
    3: 'third quadrant'
    4: 'fourth quadrant'
    angles provided must be in radians"""
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
