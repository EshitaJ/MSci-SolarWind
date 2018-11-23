import timeit
import plasmapy
from astropy import units as u
from SPC_Plot import *
from SPC_Plates import *
from SPC_Integrands import *


def V_A(B, n):
    """Returns Alfven speed"""
    return (plasmapy.physics.parameters.Alfven_speed(
        B*u.T,
        n*u.m**-3,
        ion="p+")) / (u.m / u.s)


Fractional = False
Plotting = True

va = 245531.8
start = timeit.default_timer()

# Current in the Faraday cup in a given velocity band
if Fractional:
    I_k = spi.tplquad(integrand_I,
                      -lim, lim,
                      lambda x: -lim, lambda x: lim,
                      lambda x, y: band_low, lambda x, y: band_high,
                      args=(va, constants["n"]))
    V_k = spi.tplquad(integrand_V,
                      -lim, lim,
                      lambda x: -lim, lambda x: lim,
                      lambda x, y: band_low, lambda x, y: band_high,
                      args=(va, constants["n"]))
    W_k = spi.tplquad(integrand_W,
                      -lim, lim,
                      lambda x: -lim, lambda x: lim,
                      lambda x, y: band_low, lambda x, y: band_high,
                      args=(va, constants["n"]))
    A_k = spi.tplquad(integrand_plate,
                      -lim, lim,
                      lambda x: -lim, lambda x: lim,
                      lambda x, y: band_low, lambda x, y: band_high,
                      args=(va, constants["n"], 1))
    B_k = spi.tplquad(integrand_plate,
                      -lim, lim,
                      lambda x: -lim, lambda x: lim,
                      lambda x, y: band_low, lambda x, y: band_high,
                      args=(va, constants["n"], 2))
    C_k = spi.tplquad(integrand_plate,
                      -lim, lim,
                      lambda x: -lim, lambda x: lim,
                      lambda x, y: band_low, lambda x, y: band_high,
                      args=(va, constants["n"], 3))
    D_k = spi.tplquad(integrand_plate,
                      -lim, lim,
                      lambda x: -lim, lambda x: lim,
                      lambda x, y: band_low, lambda x, y: band_high,
                      args=(va, constants["n"], 4))
    stop1 = timeit.default_timer()

    print("V_k: ", V_k[0], I_k[0])  # -1e-17  # (U-D)/(U+D)
    print("W_k: ", W_k[0], I_k[0])  # -1e-17  # (L-R)/(L+R)
    print("A_k: ", A_k[0]/I_k[0])  # 0.58 # U,R
    print("B_k: ", B_k[0]/I_k[0])  # 0.258  # U,L
    print("C_k: ", C_k[0]/I_k[0])  # 0.258  # D,L
    print("D_k: ", D_k[0]/I_k[0])  # 1.55  # D,R
    print("Temperature: ", constants["T_z"])
    print("time taken: ", stop1-start, "s")

if Plotting:
    Plot(True, 700, 450)
    Plot(False, 940, 450)

    stop = timeit.default_timer()
    print("time taken: ", stop-start, "s")

    plt.legend()
    plt.xlabel(r"$V_z\ \rm{(km/s)}$")
    plt.ylabel("Current (nA)")
    plt.show()
