#import timeit
#import plasmapy
#from astropy import units as u
#from SPC_Plot import *
#from SPC_Plates import *
#from SPC_Integrands import *
from VDF_copy import *


class SPC_class:

    def __init__(self):
        self.data_generated = False
        self.meshgrid_points = None
        self.data = None

    def gen_data(self,
                 core_fraction,
                 v_perp_min, v_perp_max,
                 v_par_min, v_par_max,
                 T_perp, T_par,
                 meshgrid_points=500):

        data = VDF(T_perp=T_perp, T_par=T_par)
        data.gen_3D(core_fraction=core_fraction,
                    meshgrid_points=meshgrid_points,
                    v_perp_min=v_perp_min,
                    v_perp_max=v_perp_max,
                    v_par_min=v_par_min,
                    v_par_max=v_par_max,
                    )
        self.data = data
        self.data_generated = True

    def rotate_data(self,
                    phi,
                    theta,
                    psi=0,
                    core_b_speed=None,
                    core_ecliptic_speed=None):

        assert self.data_generated, "You need to generate data with gen_data() method first"

        self.data.B_ecliptic_transformation(self,
                                            phi=phi,
                                            theta=theta,
                                            psi=psi,
                                            core_b_speed=core_b_speed,
                                            core_ecliptic_speed=core_ecliptic_speed)

    def fractional_discrete(self, xy_lim=0, band_low=0, band_high=0):
        x, y, z, c = self.data.VDF_3D
        #print(x, y, z, c)
        #print(x.shape, y.shape, z.shape, c.shape)
        flat_x = x.flatten()
        flat_y = y.flatten()
        flat_z = z.flatten()
        print(flat_x)
        print(np.trapz(c, z))
        #print(np.trapz(np.trapz(c, x), y))
        n = np.trapz(np.trapz(np.trapz(c, np.unique(x)), np.unique(y)), np.unique(z))
        I = n * cst.e
        print(n)


    def fractional(self, xylim, band_low, band_high):
        pass

    @staticmethod
    def Area(vz, vy, vx):
        """Effective area depends on the elevation_angle
        which is a function of velocity"""
        A_0 = 1.36e-4  # m^2
        v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        elevation_angle = np.degrees(np.arctan(vz / v) - np.pi / 2)

        # can't have negative area
        A1 = 1 - (1.15 * elevation_angle) / 90
        A2 = 1 - (4.5 * (elevation_angle - 24)) / 90
        minimum = min(A1, A2)
        area = A_0 * minimum  # m^2
        return max(area, 0)

    @staticmethod
    def detector(theta_x, theta_y, plate):
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
            return (3 * np.pi / 2 - x - y + (np.tan(x) + 2 / np.tan(y)) * (np.cos(x)) ** 2
                    + 3 * np.sin(2 * y) / 2 - 2 * np.cos(x) * np.cos(y)) / (2 * np.pi)
        elif plate == 2:
            return (x + (np.pi / 2) - y - (np.tan(x) + 2 / np.tan(y)) * (np.cos(x)) ** 2
                    - 0.5 * np.sin(2 * y) + 2 * np.cos(x) * np.cos(y)) / (2 * np.pi)
        elif plate == 3:
            return (x - (np.pi / 2) + y - (np.tan(x) - 2 / np.tan(y)) * (np.cos(x)) ** 2
                    + 0.5 * np.sin(2 * y) - 2 * np.cos(x) * np.cos(y)) / (2 * np.pi)
        elif plate == 4:
            return (y + (np.pi / 2) - x - (np.tan(y) + 2 / np.tan(x)) * (np.cos(y)) ** 2
                    - 0.5 * np.sin(2 * x) + 2 * np.cos(y) * np.cos(x)) / (2 * np.pi)

    @staticmethod
    def H(theta):
        """Function returning relative area on either side of the SPC;
        theta is angle from normal to SPC"""
        m = 1.51  # Value for SPC from alice's  report
        x = np.arccos(m * np.tan(theta))
        return (1 + (2 / np.pi) * ((np.sin(x) * np.cos(x)) - x))


if __name__ == '__main__':
    test = SPC_class()
    test.gen_data(0.8, -1e6, 1e6, -1e6, 1e6, 4e5, 14e5, 3)
    test.fractional()