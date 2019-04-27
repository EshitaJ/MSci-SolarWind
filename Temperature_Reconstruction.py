import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SPC
import SPC_Fits as spcf
import Global_Variables as gv
import itertools
import scipy.constants as cst
import multiprocessing as mp
import sys
# import sys

# import seaborn as sns
# sns.set()

# filename = "./Data/%s%s%s_N_%g_Field_%s" \
#     % ("Perturbed_" if gv.perturbed else "",
#         "final_",
#         "total" if gv.total else "core", gv.N, gv.Rot)
#
#
# band_centres = np.genfromtxt("./Data/Band_Centres_%s_%s_%s.csv"
#                              % ("Energy" if gv.E_plot else "Velocity",
#                                 "N_%g" % gv.N, "Field_%s" % gv.Rot))
#
# data1 = np.genfromtxt('%s_quad_1.csv' % filename)
# data2 = np.genfromtxt('%s_quad_2.csv' % filename)
# data3 = np.genfromtxt('%s_quad_3.csv' % filename)
# data4 = np.genfromtxt('%s_quad_4.csv' % filename)

qd1, qd2, qd3, qd4, total, band_centres = SPC.main(2.4e5, 1.7e5, "final_", False)

# total_data = data1 + data2 + data3 + data4

# fit_array = np.linspace(np.min(band_centres), np.max(band_centres), gv.N)

# mu1_guess = 700.2  # gv.v_sw[2] / 1e3
# mu2_guess = 940  # gv.beam_v[2] / 1e3
# variance_guess = 41.9


def gauss(x, N, mu, sg):
    return N * np.exp(-((x - mu) / (np.sqrt(2) * sg))**2)


# gaussfit = gauss(band_centres, np.max(total_data), mu1_guess, variance_guess)

# plt.plot(band_centres, total_data, 'o', label="data")
# plt.plot(band_centres, gaussfit, '--', label="gauss")
# plt.legend()
# plt.show()
# total = spcf.Fit(gv.E_plot, band_centres, total_data, fit_array, "",
#          band_centres[np.argmax(total_data)],
#          variance_guess, np.max(total_data))
# qd1 = spcf.Fit(gv.E_plot, band_centres, data1, fit_array, "",
#          band_centres[np.argmax(data1)],
#          variance_guess, np.max(data1))
# qd2 = spcf.Fit(gv.E_plot, band_centres, data2, fit_array, "",
#          band_centres[np.argmax(data2)],
#          variance_guess, np.max(data2))
# qd3 = spcf.Fit(gv.E_plot, band_centres, data3, fit_array, "",
#          band_centres[np.argmax(data3)],
#          variance_guess, np.max(data3))
# qd4 = spcf.Fit(gv.E_plot, band_centres, data4, fit_array, "",
#          band_centres[np.argmax(data4)],
#          variance_guess, np.max(data4))
#
#
# n1, mu1, sg1 = qd1
# n2, mu2, sg2 = qd2
# n3, mu3, sg3 = qd3
# n4, mu4, sg4 = qd4
# radial_temp = ((total[2] * 1e3)**2) * (cst.m_p / cst.k)
# print("radial_temp: ", radial_temp)


def cost_func(t_perp_guess, t_par_guess):
    comment = "Minimisation_Test_"
    # get estimated values
    load = False
    p1, p2, p3, p4, tf, band_centres = SPC.main(t_perp_guess, t_par_guess, comment, load)
    v_guess = ((cst.k * t_par_guess) / cst.m_p)**0.5 / 1e3
    # p1 = spcf.Fit(gv.E_plot, band_centres, q1, fit_array,
    #          band_centres[np.argmax(q1)], v_guess, np.max(q1))
    # p2 = spcf.Fit(gv.E_plot, band_centres, q2, fit_array,
    #          band_centres[np.argmax(q2)], v_guess, np.max(q2))
    # p3 = spcf.Fit(gv.E_plot, band_centres, q3, fit_array,
    #          band_centres[np.argmax(q3)], v_guess, np.max(q3))
    # p4 = spcf.Fit(gv.E_plot, band_centres, q4, fit_array,
    #          band_centres[np.argmax(q4)], v_guess, np.max(q4))
    n1_estimate, mu1_estimate, sg1_estimate = p1
    n2_estimate, mu2_estimate, sg2_estimate = p2
    n3_estimate, mu3_estimate, sg3_estimate = p3
    n4_estimate, mu4_estimate, sg4_estimate = p4
    # print("tot: ", t_perp_guess, t_par_guess, I_tot)
    # calculate cost
    mu_cost = (((mu1 - mu1_estimate) / mu1)**2 + ((mu2 - mu2_estimate) / mu2)**2 \
    + ((mu3 - mu3_estimate) / mu3)**2 + ((mu4 - mu4_estimate) / mu4)**2)
    sg_cost = (((sg1 - sg1_estimate) / sg1)**2 + ((sg2 - sg2_estimate) / sg2)**2 \
    + ((sg3 - sg3_estimate) / sg3)**2 + ((sg4 - sg4_estimate) / sg4)**2)
    n_cost = ((n1 - n1_estimate) / n1)**2 + ((n2 - n2_estimate) / n2)**2 \
    + ((n3 - n3_estimate) / n3)**2 + ((n4 - n4_estimate) / n4)**2

    cost = mu_cost + sg_cost + n_cost
    print("cost: ", mu_cost, sg_cost, n_cost, cost)
    return cost


def cost_func_wrapper(args):
    t_perp = args[0]
    t_para = args[1]
    try:
        return cost_func(t_perp, t_para)
    except RuntimeError:
        return 0


def heatmap():
    temps = np.linspace(1e5, 3e5, 12).tolist()
    temp_combos = list(itertools.product(temps, temps))
    # indices = list(range(len(temp_combos)))
    # perp_array = np.linspace(0.5e5, 4e5, 1e2)
    # par_array = np.linspace(0.5e5, 4e5, 1e2)
    F = np.zeros((len(temps), len(temps)))

    nsamples = 2 ** 16
    cmap = plt.cm.get_cmap('Blues', nsamples)
    newcolors = cmap(np.linspace(0, 1, nsamples))
    newcolors[0] = [1.0, 1.0, 1.0, 1.0]
    newcmp = matplotlib.colors.ListedColormap(newcolors)

    with mp.Pool() as pool:
        for i, c in enumerate(pool.imap(cost_func_wrapper, temp_combos)):
            F.flat[i] = c
            print("\033[92mCompleted %d of %d\033[0m" % (i + 1, len(temp_combos)))
            if c == 0:
                print("\033[91mFailed to converge sensibly for %d\033[0m" % (i + 1))

    F[F == 0] = np.amax(F)
    np.savetxt('%s_%s_F_nreco.csv' % (len(temps), gv.Rot), F)


    # if len(sys.argv) != 3:
    #     print("Please specify a resolution")
    #     print("Usage: %s <resolution> <B field>" % tuple(sys.argv[1:]))
    #     sys.exit(1)
    #
    # F = np.genfromtxt('%s_%s_F.csv' % (sys.argv[1], sys.argv[2]))
    # # cmap = plt.cm.get_cmap('Blues', 2**16)
    #
    # min_t = 1e5
    # max_t = 3e5
    # nsamples = int(sys.argv[1])
    expected_min = (170000, 240000)
    #
    # temps = np.linspace(min_t, max_t, nsamples).tolist()
    min_arg = np.unravel_index(np.argmin(F), F.shape)
    # print("Minimum value in mappable: %g, %g" % min_arg)
    print("True minimum temperature:  %g, %g" % expected_min)

    temp_arg = (0.5 + np.array(min_arg)) * ((max_t - min_t) / nsamples) + min_t
    print("Est. minimum temperature: %g, %g" % tuple(temp_arg[::-1].tolist()))


    f, ax = plt.subplots()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    fig = plt.figure("cost function map")
    plt.plot(170000, 240000, 'ro', label='True Minimum')
    plt.plot(temp_arg[1], temp_arg[0], 'kx', label='Estimated Minimum \n at (%g K, %g K)' % tuple(temp_arg[::-1].tolist()))
    plt.figure("decoy")
    mappable = fig.gca().imshow(F, extent=[min(temps), max(temps),
                                           min(temps), max(temps)],
                                cmap=cmap, norm=matplotlib.colors.LogNorm(np.amin(F), np.amax(F)), origin="lower")
    fig.colorbar(mappable, label="Cost function")
    fig.gca().set_xlabel("$T_{\parallel}$ (K)")
    fig.gca().set_ylabel("$T_{\perp}$ (K)")
    fig.gca().legend(title="B field at (%.02f$^\\circ$, %.02f$^\\circ$)"
               "\n from SPC normal"
               "\n True $T_{\parallel}$ at 170000 K"
               "\n True $T_{\perp}$ at 240000 K"
               % (np.degrees(np.arctan(gv.B[0]/gv.B[2]))
               ,  np.degrees(np.arctan(gv.B[1]/gv.B[2]))),
               loc='center left', bbox_to_anchor=(1.5, 0.5))
    # plt.show()
    fig.savefig("./Heatmaps/%s_nreco_plot.png" % len(temps), bbox_inches="tight")


def cost_func_1D():
    temperatures = np.linspace(1e5, 4e5, 12)
    M = []
    T_list = []
    with mp.Pool() as pool:
        # t_perp_list = temperatures + 100000
        t_par_list = np.ones(len(temperatures)) * radial_temp # temperatures - 100000
        for i, c in enumerate(pool.imap(cost_func_wrapper, zip(temperatures, t_par_list))):
        # for i, c in enumerate(pool.imap(cost_func_wrapper, zip(t_perp_list, temperatures))):
            print("\033[92mCompleted %d of %d\033[0m" % (i + 1, len(temperatures)))
            if c == 0:
                print("\033[91mFailed to converge sensibly for %d\033[0m" % (i + 1))
            else:
                M.append(c)
                T_list.append(temperatures[i])
    fig, ax = plt.subplots()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    fig3 = plt.figure("T par cost function")
    fig3.gca().plot(T_list, M, marker='x')
    fig3.gca().set_xlabel(r"$T_{\parallel}$ (K)")
    fig3.gca().set_ylabel("Cost function")
    fig3.legend(title="B field at (%.01f$^\\circ$, %.01f$^\\circ$)"
               "\n from SPC normal"
               "\n $T_{\perp}  = T_{\parallel} + 100000$"
               % (np.degrees(np.arctan(gv.B[0]/gv.B[2]))
               ,  np.degrees(np.arctan(gv.B[1]/gv.B[2]))),
               loc='center left', bbox_to_anchor=(1, 0.5))
    fig3.savefig("test_Par_1D_cost_func.png", bbox_inches="tight")


def grad_descent(coeff, t_estimate):

    def grad_cost(delta, t):
        with mp.Pool(3) as pool:
            t_perp_est, t_par_est = t
            print("perp, par: ", t_perp_est, t_par_est)
            delta_t_perp = t_perp_est + delta*t_perp_est
            delta_t_par = t_par_est + delta*t_par_est
            print("Delta: ", delta_t_perp, delta_t_par)

            results = tuple(pool.imap(cost_func_wrapper, [
                (t_perp_est, t_par_est),
                (delta_t_perp, t_par_est),
                (t_perp_est, delta_t_par)
            ]))
            cf, cf_perp, cf_par = results

        grad_cost_perp = (cf_perp - cf) / (delta * t_perp_est)
        grad_cost_par = (cf_par - cf) / (delta * t_par_est)
        grad = np.array([grad_cost_perp, grad_cost_par])
        print("grad: ", grad)
        return grad

    T_0 = np.array([t_estimate, t_estimate])  # perp, par
    T_old = T_0
    T_new = T_old
    T_list_perp = [T_0[0]]
    T_list_par = [T_0[1]]
    error = np.inf
    iteration = 0
    iter_list = [iteration]
    epsilon = 2e-6

    while error > epsilon:
        cost_grad_vec = grad_cost(0.1, T_new)
        print("BEFORE T_old, T_new:", T_old, T_new)
        T_new, T_old = (T_new - coeff * cost_grad_vec, T_new)
        print("AFTER  T_old, T_new:", T_old, T_new)
        T_list_perp.append(T_new[0])
        T_list_par.append(T_new[1])

        error = np.linalg.norm((T_new - T_old)**2) / np.linalg.norm((T_old)**2)
        iteration += 1
        print("error: ", error)
        print("iteration: ", iteration)
        print("perp, par: ", T_new)
        print("cost_grad_vec", cost_grad_vec)
        print("change", coeff * cost_grad_vec)
        print("predicted new T_new", coeff * cost_grad_vec)
        print("")
        iter_list.append(iteration)

    print("final perp, par: ", T_new)
    fig2 = plt.figure("Iterative Perp Temperature Reconstruction")
    fig3 = plt.figure("Iterative Par Temperature Reconstruction")
    fig3.gca().plot(iter_list, T_list_par, marker='x', label=r"$T_{\parallel}$")
    fig2.gca().plot(iter_list, T_list_perp, marker='x', label=r"$T_{\perp}$")
    fig2.gca().plot([0, np.max(iter_list)], [2.4e5, 2.4e5], 'b--', label=r"$True T_{\perp}$")
    fig3.gca().plot([0, np.max(iter_list)], [1.7e5, 1.7e5], 'b--', label=r"$True T_{\parallel}$")
    fig2.gca().set_xlabel("Number of iterations")
    fig3.gca().set_xlabel("Number of iterations")
    fig2.gca().set_ylabel("Reconstructed Temperature (K)")
    fig3.gca().set_ylabel("Reconstructed Temperature (K)")
    fig2.gca().legend()
    fig3.gca().legend()
    fig2.savefig("perp_temp_recon_NonR.png", bbox_inches="tight")
    fig3.savefig("par_temp_recon_NonR.png", bbox_inches="tight")
    return(T_new)


# grad_descent(np.array([1e9, 1e8]), radial_temp)
# heatmap()
# cost_func_1D()
