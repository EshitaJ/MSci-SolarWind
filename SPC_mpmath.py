import timeit
import plasmapy
from astropy import units as u
from SPC_Plot_mpmath import *
from SPC_Plates_mpmath import *
from SPC_Integrands_mpmath import *
import mpmath


def V_A(B, n):
    """Returns Alfven speed"""
    return (plasmapy.physics.parameters.Alfven_speed(
        B*u.T,
        n*u.m**-3,
        ion="p+")) / (u.m / u.s)


Fractional = True
Plotting = False
Norm = False

start = timeit.default_timer()

# Current in the Faraday cup in a given velocity band
if Fractional:
    I_k = mpmath.quad(lambda z, y, x: integrand_I(z, y, x, va, constants["n"]),
                      [band_low, band_high], [-lim, lim], [-lim, lim])
    V_k = mpmath.quad(lambda z, y, x: integrand_V(z, y, x, va, constants["n"]),
                      [band_low, band_high], [-lim, lim], [-lim, lim])
    W_k = mpmath.quad(lambda z, y, x: integrand_W(z, y, x, va, constants["n"]),
                      [band_low, band_high], [-lim, lim], [-lim, lim])
    A_k = mpmath.quad(lambda z, y, x: integrand_plate(z, y, x, va, constants["n"], 1),
                      [band_low, band_high], [-lim, lim], [-lim, lim])
    B_k = mpmath.quad(lambda z, y, x: integrand_plate(z, y, x, va, constants["n"], 2),
                      [band_low, band_high], [-lim, lim], [-lim, lim])
    C_k = mpmath.quad(lambda z, y, x: integrand_plate(z, y, x, va, constants["n"], 3),
                      [band_low, band_high], [-lim, lim], [-lim, lim])
    D_k = mpmath.quad(lambda z, y, x: integrand_plate(z, y, x, va, constants["n"], 4),
                      [band_low, band_high], [-lim, lim], [-lim, lim])
    stop = timeit.default_timer()

    print("V_k: ", V_k/I_k)  # -1e-17  # (U-D)/(U+D)
    print("W_k: ", W_k/I_k)  # -1e-17  # (L-R)/(L+R)
    print("A_k: ", A_k/I_k)  # 0.58 # U,R
    print("B_k: ", B_k/I_k)  # 0.258  # U,L
    print("C_k: ", C_k/I_k)  # 0.258  # D,L
    print("D_k: ", D_k/I_k)  # 1.55  # D,R
    print("Temperature: ", constants["T_z"])
    print("time taken: ", stop-start, "s")

if Plotting:
    Plot(True, 700, 450)
    Plot(False, 940, 450)

    stop = timeit.default_timer()
    print("time taken: ", stop-start, "s")

    plt.legend()
    plt.xlabel(r"$V_z\ \rm{(km/s)}$")
    plt.ylabel("Current (nA)")
    plt.show()

if Norm:
    """Normalisation check: core + beam should equal number density, n
    We don't expect normalisation to hold here if SPC can only measure
    part of the beam"""
    '''core = spi.tplquad(BiMax,
                       -lim, lim,
                       lambda x: -lim, lambda x: lim,
                       lambda x, y: -lim, lambda x, y: lim,
                       args=(va, True, constants["n"]))
    beam = spi.tplquad(BiMax,
                       -lim, lim,
                       lambda x: -lim, lambda x: lim,
                       lambda x, y: -lim, lambda x, y: lim,
                       args=(va, False, constants["n"]))'''

    def f(z, y, x):
        return BiMax(z, y, x, va, True, constants["n"])

    def g(z, y, x):
        return BiMax(z, y, x, va, False, constants["n"])

    core = mpmath.quad(f,
                       [-lim, lim], [-lim, lim], [-lim, lim])
    beam = mpmath.quad(g,
                       [-lim, lim], [-lim, lim], [-lim, lim])
    print("Core: ", core)
    print("Beam: ", beam)
    print("Total: ", core + beam)
    print("n: ", constants["n"])
    stop = timeit.default_timer()
    print("time taken: ", stop-start, "s")
