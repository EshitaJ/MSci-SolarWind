import timeit
import SPC_Plot as spcp
import matplotlib.pyplot as plt
import numpy as np
import Global_Variables as gv
import scipy.constants as cst


def main(perp, par, comment, load):

    # start = timeit.default_timer()

    vcore_peak = gv.v_sw[2] / 1e3
    vbeam_peak = gv.beam_v[2] / 1e3
    Ecore_pk = (0.5 * cst.m_p * (vcore_peak*1e3)**2) / gv.J
    Ebeam_pk = (0.5 * cst.m_p * (vbeam_peak*1e3)**2) / gv.J
    guess = (0.5 * cst.m_p * (40*1e3)**2) / gv.J

    core_peak = Ecore_pk if gv.E_plot else vcore_peak
    beam_peak = Ebeam_pk if gv.E_plot else vbeam_peak
    # print("core, beam: ", core_peak, beam_peak)
    sigma = (guess*2)**2 if gv.E_plot else 40

    par_dict = {
             'n': "%.2E" % gv.constants['n'],
             'T_perp': "%.1E" % perp,
             'T_par': "%.1E" % par,
             'B strength': gv.constants["B"],
             'Bx': gv.B[0],
             'By': gv.B[1],
             'Bz': gv.B[2],
             'Bxz': round(np.degrees(np.arctan(gv.B[0]/gv.B[2])), 2),
             'Byz': round(np.degrees(np.arctan(gv.B[1]/gv.B[2])), 2),
             'Bulkx': round(gv.v_sw[0]/1e3, 2),
             'Bulky': round(gv.v_sw[1]/1e3, 2),
             'Bulkz': round(gv.v_sw[2]/1e3, 2),
             'Beamx': round(gv.beam_v[0]/1e3, 2),
             'Beamy': round(gv.beam_v[1]/1e3, 2),
             'Beamz': round(gv.beam_v[2]/1e3, 2),
             'SCx': gv.v_sc[0]/1e3,
             'SCy': gv.v_sc[1]/1e3,
             'SCz': gv.v_sc[2]/1e3,
             'Alfven speed': gv.va/1e3,
             'N': gv.N,
             'Rot': gv.Rot,
             'Perturbed': gv.perturbed,
             'dB_x': gv.dB[0],
             'dB_y': gv.dB[1],
             'dB_z': gv.dB[2],
             'dv_x': gv.dv[0],
             'dv_y': gv.dv[1],
             'dv_z': gv.dv[2],
             'Total': gv.total,
             'Core': gv.Core,
             'Core fraction': gv.core_fraction
             }
    data = spcp.Plot(
              gv.E_plot, gv.total,
              gv.Core, gv.Plates,
              core_peak, beam_peak,
              sigma, perp, par, par_dict, load, comment, num=spcp.N)

    # stop = timeit.default_timer()
    # print("time taken: ", (stop - start) / 60.0, "mins")
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("perp", help="Perpendicular SW temperature, in Kelvin",
                        type=float)
    parser.add_argument("par", help="Parallel SW temperature, in Kelvin",
                        type=float)
    parser.add_argument("comment", help="Comment or notes about the specific run",
                        type=str)
    parser.add_argument("--load", help="Loads data if True, creates if False",
                        action='store_true')
    args = parser.parse_args()

    main(args.perp, args.par, args.comment, args.load)

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
