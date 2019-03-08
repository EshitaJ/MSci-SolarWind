import timeit
import SPC_Plot as spcp
import matplotlib.pyplot as plt
import numpy as np
import Global_Variables as gv
import scipy.constants as cst


Plotting = True

start = timeit.default_timer()

if Plotting:
    vcore_peak = gv.v_sw[2] / 1e3
    vbeam_peak = gv.beam_v[2] / 1e3
    Ecore_pk = (0.5 * cst.m_p * (vcore_peak*1e3)**2) / gv.J
    Ebeam_pk = (0.5 * cst.m_p * (vbeam_peak*1e3)**2) / gv.J
    guess = (0.5 * cst.m_p * (40*1e3)**2) / gv.J

    core_peak = Ecore_pk if gv.E_plot else vcore_peak
    beam_peak = Ebeam_pk if gv.E_plot else vbeam_peak
    print("core, beam: ", core_peak, beam_peak)
    var = (guess*2)**2 if gv.E_plot else 30

    spcp.Plot(
              gv.E_plot, gv.total,
              gv.Core, gv.Plates,
              core_peak, beam_peak,
              var, num=spcp.N)

    stop = timeit.default_timer()
    print("time taken: ", (stop - start) / 60.0, "mins")
    plt.show()


# def RBM(x, y):
#     core = spcp.rotatedMW(gv.va, y, x, 0, True, gv.constants["n"])
#     beam = spcp.rotatedMW(gv.va, y, x, gv.va, False, gv.constants["n"])
#     return core + beam


# lim1 = 5e5
# vx = np.linspace(-lim1, lim1, 100)
# vy = np.linspace(-lim1, lim1, 100)
# Vx, Vy = np.meshgrid(vx, vy)  # SC frame
# Z = RBM(Vx, Vy)  # B frame
#
# plt.contour(Vx/1e3, Vy/1e3, Z)
# plt.xlabel("$V_x$ (km/s)")
# plt.ylabel("$V_y$ (km/s)")
# plt.quiver(gv.B[0], gv.B[1])
# plt.gca().set_aspect("equal")
#
# plt.show()
