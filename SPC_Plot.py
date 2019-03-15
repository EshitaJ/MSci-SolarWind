import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.constants as cst
from SPC_Fits import *
from Global_Variables import *
from scipy.stats import norm
import Global_Variables as gv
from SPC_Plates import *
from VDF import *
sns.set()


def current_vdensity(vz, vy, vx, v, is_core, n):
    df = rotatedMW(vz, vy, vx, v, is_core, n, core_fraction)
    return cst.e * np.sqrt(vz**2 + vy**2 + vx**2) * Area(vz, vy, vx) * df


def integrand_plate(vz, theta, phi, v_alf, n):
    """integral returns the current in a given detector plate"""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    v = vz / cos_theta
    vx = v * sin_theta * np.cos(phi)
    vy = v * sin_theta * np.sin(phi)
    th_x = np.arctan(vx / vz)
    th_y = np.arctan(vy / vz)

    # SPC has an angular range of +/- 30 degrees and so only sees these angles
    angular_range = pi / 6

    if np.abs(th_x) < angular_range and np.abs(th_y) < angular_range:
        jacobian = (v ** 2) * sin_theta
        core = current_vdensity(vz, vy, vx, v_alf, True, n)
        if total:
            beam = current_vdensity(vz, vy, vx, v_alf, False, n)
            cvd = core + beam
        else:
            cvd = core

        return cvd * jacobian
    else:
        return 0


def Signal_Count(bounds, is_core, plates, plate):
    output = []
    low = bounds[:-1]
    high = bounds[1:]

    def integration(low_bound, high_bound):
        if plates:
            if plate == 1:
                low_phi = -pi / 2
                high_phi = -pi
                low_theta = 0.0
                high_theta = -pi / 2
            elif plate == 2:
                low_phi = -pi
                high_phi = -pi * (3 / 2)
                low_theta = -pi / 2
                high_theta = -pi
            elif plate == 3:
                low_phi = -pi * (3 / 2)
                high_phi = -pi * 2
                low_theta = 0
                high_theta = -pi / 2
            elif plate == 4:
                low_phi = 0
                high_phi = -pi / 2
                low_theta = -pi / 2
                high_theta = -pi

            I_k = spi.tplquad(integrand_plate,
                              low_phi, high_phi,
                              lambda phi: low_theta, lambda phi: high_theta,
                              lambda phi, theta: low_bound,  # vz
                              lambda phi, theta: high_bound,
                              args=(va, constants["n"]))
        else:
                I_k = spi.tplquad(current_vdensity,
                                  -lim, lim,
                                  lambda x: -lim, lambda x: lim,
                                  lambda x, y: low_bound,
                                  lambda x, y: high_bound,
                                  args=(va, is_core, constants["n"]))
        return I_k

    integration = np.vectorize(integration)
    output = np.abs(integration(low, high)[0])
    return output


def Data(velocities, is_core, plates, plate):
    if plates:
        if plate == 1:
            if load:
                signal = np.genfromtxt('%s_quad_1.csv' % gv.filename)
            else:
                signal = Signal_Count(velocities, is_core, plates, plate) * 1e9

                np.savetxt('%s_quad_1.csv' % gv.filename, signal)
                print("Saved, quad 1 data")
        if plate == 2:
            if load:
                signal = np.genfromtxt('%s_quad_2.csv' % gv.filename)

            else:
                signal = Signal_Count(velocities, is_core, plates, plate) * 1e9

                np.savetxt('%s_quad_2.csv' % gv.filename, signal)
                print("Saved, quad 2 data")
        if plate == 3:
            if load:
                signal = np.genfromtxt('%s_quad_3.csv' % gv.filename)

            else:
                signal = Signal_Count(velocities, is_core, plates, plate) * 1e9

                np.savetxt('%s_quad_3.csv' % gv.filename, signal)
                print("Saved, quad 3 data")
        if plate == 4:
            if load:
                signal = np.genfromtxt('%s_quad_4.csv' % gv.filename)

            else:
                signal = Signal_Count(velocities, is_core, plates, plate) * 1e9

                np.savetxt('%s_quad_4.csv' % gv.filename, signal)
                print("Saved, quad 4 data")

    else:
        if is_core:
            if load:
                signal = np.genfromtxt('%s_core.csv' % gv.filename)
            else:
                signal = Signal_Count(velocities, True, False, plate) * 1e9

                np.savetxt('%s_core.csv' % gv.filename, signal)
                print("Saved, core")
        else:
            if load:
                signal = np.genfromtxt('%s_beam.csv' % gv.filename)
            else:
                signal = Signal_Count(velocities, False, False, plate) * 1e9

                np.savetxt('%s_beam.csv' % gv.filename, signal)
                print("Saved, beam")
    return signal


def Plot(E_plot, plot_total, is_core, plates,
         mu1_guess, mu2_guess, variance_guess, num=50):
    """ @ E_plot=True plots current against Energy, and otherwise plots
    current against z-velocity;
    @ plot_total=True plots the total core+beam vdf with a double gaussian fit,
    and otherwise plots and fits either core or beam vdf
    (function needs to be called once for core and once for beam if =False);
    @ is_core=True (relevant iff plot_total=False) calculates and plots core,
    and otherwise beam;
    @ plates=True plots the current in each of the 4 plates
    @ mu1_guess and mu2_guess are used as estimates of
    core and beam peaks respectively for fitting;
    @ variance_guess is estimate of variance for fitting (for now,
    assume core and beam distributions have same width)
    @ num is number of bins, default is set to 50
    """

    # unequal bin widths in velocity as potential is what is varied
    # for now assume equal bin widths in potential, but can change later
    potential = np.linspace(100, 8000, int(num))
    vz_m = np.sqrt((2 * potential * J) / cst.m_p)  # z velocity in m/s
    # vz_m = np.linspace(650000, 750000, int(num))
    vz_k = vz_m / 1e3  # velocity in km/s for plotting purposes

    v_band_centres = (vz_k[:-1] + vz_k[1:]) / 2.0
    fit_array_v = np.linspace(np.min(vz_k), np.max(vz_k), int(1e3))

    E_band_centres = (potential[:-1] + potential[1:]) / 2.0
    fit_array_E = np.linspace(np.min(potential), np.max(potential), int(1e3))

    if E_plot:
        band_centres = E_band_centres
        band_width = np.diff(potential)
        fit_array = fit_array_E
        dvdE = 1  # np.sqrt(cst.m_p / (2*band_centres*gv.J))
        print("dvDe: ", dvdE)
    else:
        band_centres = v_band_centres
        band_width = np.diff(vz_k)
        fit_array = fit_array_v

    np.savetxt("./Data/Band_Centres_%s_%s_%s.csv"
               % ("Energy" if E_plot else "Velocity",
                  "N_%g" % N, "Field_%s" % Rot),
               band_centres
               )
    print('Saved, band centres')

    fig, ax = plt.subplots()

    if plates:
        quad1 = Data(vz_m, True, True, 1)
        quad2 = Data(vz_m, True, True, 2)
        quad3 = Data(vz_m, True, True, 3)
        quad4 = Data(vz_m, True, True, 4)

        quad1 = quad1 * dvdE if E_plot else quad1
        quad2 = quad2 * dvdE if E_plot else quad2
        quad3 = quad3 * dvdE if E_plot else quad3
        quad4 = quad4 * dvdE if E_plot else quad4

        U = quad1 + quad2
        D = quad3 + quad4
        L = quad2 + quad3
        R = quad1 + quad4
        total_quads = quad1 + quad2 + quad3 + quad4
        V = (U-D) / (U+D)
        W = (R-L) / (R+L)
        V_Tot = V*total_quads
        W_Tot = W*total_quads

        """Sloppy estimate of fraction of population in core"""
        cut_off = 840
        core_guess = total_quads[band_centres < cut_off]
        beam_guess = total_quads[band_centres > cut_off]
        fraction_guess = np.sum(core_guess) / np.sum(total_quads)
        print("Core fraction estimate: ", fraction_guess)
        plt.figure(1)
        plt.plot(band_centres, V_Tot, 'xkcd:diarrhea', label='V * Total', marker='x')
        plt.plot(band_centres, W_Tot, 'xkcd:hot pink', label='W * Total', marker='x')
        Fit(E_plot, band_centres, V_Tot, fit_array,
            mu1_guess, variance_guess, 0.1)
        Fit(E_plot, band_centres, W_Tot, fit_array,
            mu1_guess, variance_guess, 0.1)
        plt.ylabel("Current Difference (nA)")
        plt.legend()

        plt.figure(2)
        plt.plot(band_centres, V, 'xkcd:diarrhea', label='V')
        plt.plot(band_centres, W, 'xkcd:hot pink', label='W')
        plt.plot(band_centres, quad1/total_quads, 'k-', label='Quadrant 1')
        plt.plot(band_centres, quad2/total_quads, 'r-', label='Quadrant 2')
        plt.plot(band_centres, quad3/total_quads, 'g-', label='Quadrant 3')
        plt.plot(band_centres, quad4/total_quads, 'b-', label='Quadrant 4')
        plt.ylabel("Fractional Current")
        plt.legend()
        # plt.plot([700, 700], [-1, 1], '--')
        # plt.plot(band_centres, np.ones(len(band_centres)) * 0.25, '--')

        # p = (1 + V[np.argmax(total_quads)]) / 2
        # d = (quad1 + quad2) / total_quads
        # px = (1 + W[np.argmax(total_quads)]) / 2
        # dx = (quad1 + quad4) / total_quads
        # vy_estimate = norm.ppf(p) * gv.ythermal_speed / 1e3
        # quad_estimate_vy = norm.ppf(d) * gv.ythermal_speed / 1e3
        # vx_estimate = norm.ppf(px) * gv.xthermal_speed / 1e3
        # quad_estimate_vx = norm.ppf(dx) * gv.xthermal_speed / 1e3
        # print("y average: ", np.average(vy_estimate))
        # print("x average: ", np.average(vx_estimate))

        # plt.plot(band_centres, vy_estimate, '-k', label='Using V')
        # plt.plot(band_centres, quad_estimate_vy, 'ro', label='Using quadrants 1 and 2')
        # plt.plot(band_centres, vx_estimate, '-g', label='Using W')
        # plt.plot(band_centres, quad_estimate_vx, 'bo', label='Using quadrants 1 and 4')
        # plt.plot([100, 1200], [-1, 1], '--')
        # plt.plot([100, 1200], [0, 0], '--')
        # plt.plot([700, 700], [-75, 75], '--')
        # plt.ylabel('Recovered solar wind bulk speed (km/s)')

        plt.figure(3)
        plt.plot(band_centres, quad1, 'k--', label='Quadrant 1')
        plt.plot(band_centres, quad2, 'ro', label='Quadrant 2')
        plt.plot(band_centres, quad3, 'gx', label='Quadrant 3')
        plt.plot(band_centres, quad4, 'b-', label='Quadrant 4')
        plt.ylabel("Current (nA)")
        n1 = 0.003
        n2 = 0.001

        q1 = Total_Fit(E_plot, band_centres, quad1, fit_array, False,
                       mu1_guess, mu2_guess, variance_guess, n1, n2)
        q2 = Total_Fit(E_plot, band_centres, quad2, fit_array, False,
                       mu1_guess, mu2_guess, variance_guess, n1, n2)
        q3 = Total_Fit(E_plot, band_centres, quad3, fit_array, False,
                       mu1_guess, mu2_guess, variance_guess, n1, n2)
        q4 = Total_Fit(E_plot, band_centres, quad4, fit_array, False,
                       mu1_guess, mu2_guess, variance_guess, n1, n2)
        plt.legend()

        plt.figure(4)
        plt.plot(band_centres, total_quads, 'bx', label='Total Current')
        Total_Fit(E_plot, band_centres, total_quads, fit_array, True,
            mu1_guess, mu2_guess, variance_guess, 0.09, 0.01)
        plt.ylabel("Current (nA)")
        plt.legend()
        # plt.plot(band_centres, total_quads, label='Sum of quadrants')




            # quad_estimate_vx = norm.ppf(dx) * gv.xthermal_speed / 1e3
            print("beam y average: ", np.average(vy_estimatebeam))
            print("beam x average: ", np.average(vx_estimatebeam))


    else:
        if plot_total:
            core = Data(vz_m, True, False, 1)  # velocities in m/s
            beam = Data(vz_m, False, False, 1)  # velocities in m/s
            core = core * dvdE if E_plot else core
            beam = beam * dvdE if E_plot else beam
            total = core + beam
            # print("total: ", total)

            """Sloppy estimate of fraction of population in core"""
            cut_off = 1800
            core_guess = total[band_centres > cut_off]
            beam_guess = total[band_centres < cut_off]
            fraction_guess = np.sum(core_guess) / np.sum(total)
            print("Core fraction estimate: ", fraction_guess)

            # can either plot total or beam stacked on core - both are same
            plt.bar(band_centres, core, width=band_width,
                    label="Measured core at $T_z = %g$" % constants["T_z"])
            plt.bar(band_centres, beam, width=band_width, bottom=core,
                    label="Measured beam at $T_z = %g$" % constants["T_z"])

            # Total_Fit(E_plot, band_centres, total, fit_array, True,
            #           mu1_guess, mu2_guess, variance_guess,  n1, n2)
            plt.ylabel("Current (nA)")

    xlabel = "{x}".format(x="Energy (eV)" if E_plot else "$V_z$ (km/s)")
    plt.xlabel(xlabel)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    if load:
        Bxz = gv.mydict['Bxz']
        Byz = gv.mydict['Byz']
        T_perp = gv.mydict['T_perp']
        T_par = gv.mydict['T_par']
        vx = gv.mydict['Bulkx']
        vy = gv.mydict['Bulky']
        vz = gv.mydict['Bulkz']
        bvx = gv.mydict['Beamx']
        bvy = gv.mydict['Beamy']
        bvz = gv.mydict['Beamz']
        fraction = gv.mydict['Core fraction']
    else:
        Bxz = round(np.degrees(np.arctan(gv.B[0]/gv.B[2])), 2)
        Byz = round(np.degrees(np.arctan(gv.B[1]/gv.B[2])), 2)
        T_perp = "%.1E" % constants['T_perp']
        T_par = "%.1E" % constants['T_par']
        vx = gv.v_sw[0]/1e3
        vy = gv.v_sw[1]/1e3
        vz = gv.v_sw[2]/1e3
        bvx = gv.beam_v[0]/1e3
        bvy = gv.beam_v[1]/1e3
        bvz = gv.beam_v[2]/1e3
        fraction = gv.core_fraction

    # print(fieldx, fieldy, fieldz)

    # plt.legend(title="%s B field at (%s$^\\circ$, %s$^\\circ$)"
    #            "\n from SPC normal"
    #            "\n $T_{\perp}$ = %s K"
    #            "\n $T_{\par}$ = %s K"
    #            "\n Core velocity [%s, %s, %s] km/s"
    #            "\n Recovered x bulk speed"
    #            "\n of $V_{x}$ = %.2f km/s"
    #            "\n Recovered y bulk speed"
    #            "\n of $V_{y}$ = %.2f km/s"
    #             "\n %s"
    #             "\n %s"
    #             "\n %s"
    #             "\n %s"
    #             "\n %s"
    #            % (
    #               "Perturbed" if gv.perturbed else "",
    #               Bxz, Byz,
    #               T_perp,
    #               T_par,
    #               vx, vy, vz,
    #               np.average(vx_estimate),
    #               np.average(vy_estimate),
    #               "Beam velocity [%s, %s, %s] km/s" % (
    #                 bvx, bvy, bvz) if gv.total else "",
    #               "Core fraction = %s" % (fraction) if gv.total else "",
    #               "Rough estimate = %s" % (fraction_guess) if gv.total else "",
    #               "Recovered beam x speed of $V_{x}$ = %.2f km/s" \
    #               % np.average(vx_estimatebeam) if gv.total else "",
    #               "Recovered beam y speed of $V_{y}$ = %.2f km/s" \
    #               % np.average(vy_estimatebeam) if gv.total else ""),
    #            loc='center left', bbox_to_anchor=(1, 0.5))

    return q1, q2, q3, q4
