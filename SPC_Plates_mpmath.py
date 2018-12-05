import mpmath as mpm
import scipy.constants as cst


def Detector(theta_x, theta_y, plate):
    """Returns area of beam hitting the detector plate
    1: 'first quadrant'
    2: 'second quadrant'
    3: 'third quadrant'
    4: 'fourth quadrant'
    angles provided must be in radians"""
    m = 1.51  # ratio of aperture radius to aperture vertical displacement
    x = mpm.acos(m * mpm.tan(theta_x))
    y = mpm.acos(m * mpm.tan(theta_y))

    if plate == 1:
        area1 = (3*mpm.pi/2 - x - y + (mpm.tan(x) + 2/mpm.tan(y))*(mpm.cos(x))**2
                + 3*mpm.sin(2*y)/2 - 2*mpm.cos(x)*mpm.cos(y)) / (2*mpm.pi)
        # print(area1)
        return area1
    elif plate == 2:
        area2 = (x + (mpm.pi/2) - y - (mpm.tan(x) + 2/mpm.tan(y))*(mpm.cos(x))**2
                - 0.5*mpm.sin(2*y) + 2*mpm.cos(x)*mpm.cos(y)) / (2*mpm.pi)
        # print(area2)
        return area2
    elif plate == 3:
        area3 = (x - (mpm.pi/2) + y - (mpm.tan(x) - 2/mpm.tan(y))*(mpm.cos(x))**2
                + 0.5*mpm.sin(2*y) - 2*mpm.cos(x)*mpm.cos(y)) / (2*mpm.pi)
        # print(area3)
        return area3
    elif plate == 4:
        area4 = (y + (mpm.pi/2) - x - (mpm.tan(y) + 2/mpm.tan(x))*(mpm.cos(y))**2
                - 0.5*mpm.sin(2*x) + 2*mpm.cos(y)*mpm.cos(x)) / (2*mpm.pi)
        # print(area4)
        return area4
