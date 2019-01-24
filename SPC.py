import timeit
import plasmapy
from astropy import units as u
from SPC_Plot import *
# from SPC_Plates import *

Plotting = True

start = timeit.default_timer()

if Plotting:
    Plot(False, False, True, True, 700, 900, 250, num=N)
    # Plot(False, False, False, False, 700, 900, 250)
    # Plot(True, True, True, False, 2000, 5000, 500)
    # Plot(True, False, True, False, 2000, 5000, 500)
    # Plot(True, False, False, False, 2000, 5000, 500)

    stop = timeit.default_timer()
    print("time taken: ", (stop-start)/60., "mins")
    plt.show()


def RBM(x, y):
    core = rotatedMW(700000, y, x, 0, True, constants["n"], B0)
    # beam = rotatedMW(700000+va, y, x, va, False, constants["n"], B0)
    return core  # + beam


def BM(x, y):
    core = BiMax(700000, y, x, 0, True, constants["n"])
    # beam = BiMax(0, y, x, 2*va, False, constants["n"])
    return core


if __name__ == '__main__':

    lim1 = 1e6
    x = np.linspace(-lim1, lim1, 100)
    y = np.linspace(-lim1, lim1, 100)
    X, Y = np.meshgrid(x, y)  # SC frame
    Z = RBM(X, Y)  # B frame

    plt.contour(X/1e3, Y/1e3, Z)
    plt.xlabel("$V_x$ (km/s)")
    plt.ylabel("$V_y$ (km/s)")
    plt.quiver(B0[0], B0[1])
    plt.gca().set_aspect("equal")

    plt.show()
