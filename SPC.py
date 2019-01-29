import timeit
import SPC_Plot as spcp
import matplotlib.pyplot as plt
# from SPC_Plates import *

Plotting = True

start = timeit.default_timer()

if Plotting:
    spcp.Plot(False, False, True, True, 700, 900, 250, num=spcp.N)
    # Plot(False, False, False, False, 700, 900, 250)
    # Plot(True, True, True, False, 2000, 5000, 500)
    # Plot(True, False, True, False, 2000, 5000, 500)
    # Plot(True, False, False, False, 2000, 5000, 500)

    stop = timeit.default_timer()
    print("time taken: ", (stop - start) / 60.0, "mins")
    plt.show()


def RBM(x, y):
    core = rotatedMW(700000, y, x, 0, True, constants["n"])
    # beam = rotatedMW(700000+va, y, x, va, False, constants["n"])
    return core   # + beam


# if __name__ == '__main__':
#
#     lim1 = 2e6
#     vz = np.linspace(-lim1, lim1, 100)
#     vy = np.linspace(-lim1, lim1, 100)
#     Vz, Vy = np.meshgrid(vz, vy)  # SC frame
#     Z = RBM(Vz, Vy)  # B frame
#
#     plt.contour(Vz/1e3, Vy/1e3, Z)
#     plt.xlabel("$V_x$ (km/s)")
#     plt.ylabel("$V_y$ (km/s)")
#     plt.quiver(B0[0], B0[1])
#     plt.gca().set_aspect("equal")
#
#     plt.show()
