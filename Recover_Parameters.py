import numpy as np
import Global_Variables as gv
import scipy.constants as cst

band_centres = np.genfromtxt("./Data/Band_Centres_%s_%s_%s.csv"
                             % ("Energy" if gv.E_plot else "Velocity",
                                "N_%g" % gv.N, "Field_%s" % gv.Rot))

core = np.genfromtxt("./Data/core_%s_%s_data.csv"
                     % ("N_%g" % gv.N, "Field_%s" % gv.Rot))
beam = np.genfromtxt("./Data/beam_%s_%s_data.csv"
                     % ("N_%g" % gv.N, "Field_%s" % gv.Rot))

vel = np.sqrt((2 * band_centres * gv.J)
              / cst.m_p) if gv.E_plot else band_centres * 1e3

vz = gv.v_sw[2]
v = np.linalg.norm(gv.v_sw)
area = 1.36e-4  # m^2


def number_density(core_df, beam_df, velocity, area):
    core_den = (core_df / velocity) / (1e9 * cst.e * area)
    beam_den = (beam_df / velocity) / (1e9 * cst.e * area)
    bar_den = core_den + beam_den
    number_den = np.sum(bar_den)
    print("Number density: ", number_den)
    print("n: ", gv.constants["n"])
    return number_den


number_density(core, beam, vel, area)
