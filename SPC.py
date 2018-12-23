import timeit
import plasmapy
from astropy import units as u
from SPC_Plot import *
from SPC_Plates import *
from SPC_Integrands import *
import pandas as pd

Fractional = False
Plotting = False

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
    Quadrant1 = spi.tplquad(integrand_plate,
                            -lim, lim,
                            lambda x: -lim, lambda x: lim,
                            lambda x, y: band_low, lambda x, y: band_high,
                            args=(va, constants["n"], 1))
    Quadrant2 = spi.tplquad(integrand_plate,
                            -lim, lim,
                            lambda x: -lim, lambda x: lim,
                            lambda x, y: band_low, lambda x, y: band_high,
                            args=(va, constants["n"], 2))
    Quadrant3 = spi.tplquad(integrand_plate,
                            -lim, lim,
                            lambda x: -lim, lambda x: lim,
                            lambda x, y: band_low, lambda x, y: band_high,
                            args=(va, constants["n"], 3))
    Quadrant4 = spi.tplquad(integrand_plate,
                            -lim, lim,
                            lambda x: -lim, lambda x: lim,
                            lambda x, y: band_low, lambda x, y: band_high,
                            args=(va, constants["n"], 4))
    stop1 = timeit.default_timer()

    print("V_k: ", V_k[0]/I_k[0])  # 0.0 (smaller than 1e-200) # (U-D)/(U+D)
    print("W_k: ", W_k[0]/I_k[0])  # 0.0  # (L-R)/(L+R)
    print("Quadrant1: ", Quadrant1[0]/I_k[0])  # 0.249999997 # U,R
    print("Quadrant2: ", Quadrant2[0]/I_k[0])  # 0.249999997  # U,L
    print("Quadrant3: ", Quadrant3[0]/I_k[0])  # 0.249999997  # D,L
    print("Quadrant4: ", Quadrant4[0]/I_k[0])  # 0.249999997  # D,R
    print("Norm: ",
          ((Quadrant1[0]+Quadrant2[0]+Quadrant3[0]+Quadrant4[0])/I_k[0]))
    print("Temperature: ", constants["T_z"])
    print("time taken: ", stop1-start, "s")

if Plotting:
    Plot(False, True, True, 700, 900, 250)
    # Plot(False, False, False, 700, 900, 250)
    # Plot(True, True, True, 2000, 5000, 500)
    # Plot(True, False, True, 2000, 5000, 500)
    # Plot(True, False, False, 2000, 5000, 500)

    stop = timeit.default_timer()
    print("time taken: ", stop-start, "s")
    plt.show()


def RBM(x, y):
    core = rotatedMW(700, y, x, 0, True, constants["n"], B0)
    # beam = rotatedMW(0, y, x, va, False, constants["n"], B0)
    return core


def BM(x, y):
    core = BiMax(700, y, x, 0, True, constants["n"])
    # beam = BiMax(0, y, x, 2*va, False, constants["n"])
    return core


lim1 = 1e6
x = np.linspace(-lim1, lim1, 100)
y = np.linspace(-lim1, lim1, 100)
X, Y = np.meshgrid(x, y)  # SC frame
Z = RBM(X, Y)
plt.contour(X/1e3, Y/1e3, Z)
plt.xlabel("$V_x$ (km/s)")
plt.ylabel("$V_y$ (km/s)")
stop = timeit.default_timer()
print("time taken: ", stop-start, "s")
plt.show()
