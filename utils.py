import scipy.constants as cst
import numpy as np


def BiMax(x, y, z, v_A, T_x, T_y, T_z, n, core_fraction, is_core, bulk_velocity=700000):
    if is_core:
        n_p = core_fraction * n
        z_new = z - bulk_velocity  # + bulk vel?
    else:
        n_p = (1 - core_fraction) * n
        z_new = z - v_A - bulk_velocity

    norm = n_p * (cst.m_p/(2 * np.pi * cst.k))**1.5 / np.sqrt(T_x * T_y * T_z)
    exponent = -((z_new**2/T_z) + (y**2/T_y) + (x**2/T_x)) \
        * (cst.m_p/(2 * cst.k))
    vdf = norm * np.exp(exponent)
    return vdf


def sph_to_cart(r, theta, phi):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    x = r * sin_theta * cos_phi
    y = r * sin_theta * sin_phi
    z = r * cos_theta

    return x, y, z
