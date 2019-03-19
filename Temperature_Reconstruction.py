import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SPC
import SPC_Fits as spcf
import Global_Variables as gv
import itertools
import multiprocessing as mp
import seaborn as sns
sns.set()

filename = "./Data/%s%s%s_N_%g_Field_%s" \
    % ("Perturbed_" if gv.perturbed else "",
        "Realistic_SW_",
        "total" if gv.total else "core", gv.N, gv.Rot)


band_centres = np.genfromtxt("./Data/Band_Centres_%s_%s_%s.csv"
                             % ("Energy" if gv.E_plot else "Velocity",
                                "N_%g" % gv.N, "Field_%s" % gv.Rot))

data1 = np.genfromtxt('%s_quad_1.csv' % filename)
data2 = np.genfromtxt('%s_quad_2.csv' % filename)
data3 = np.genfromtxt('%s_quad_3.csv' % filename)
data4 = np.genfromtxt('%s_quad_4.csv' % filename)
total_data = data1 + data2 + data3 + data4

fit_array = np.linspace(np.min(band_centres), np.max(band_centres), gv.N)


mu1_guess = gv.v_sw[2] / 1e3
mu2_guess = gv.beam_v[2] / 1e3
variance_guess = 40

total = spcf.Total_Fit(gv.E_plot, band_centres, total_data, fit_array, True,
                       mu1_guess, mu2_guess, variance_guess, 0.09, 0.01)
# n1 = 0.03
# n2 = 0.01
#
# q1 = spcf.Total_Fit(gv.E_plot, band_centres, data1, fit_array, False,
#                     mu1_guess, mu2_guess, variance_guess, n1, n2)
# q2 = spcf.Total_Fit(gv.E_plot, band_centres, data2, fit_array, False,
#                     mu1_guess, mu2_guess, variance_guess, n1, n2)
# q3 = spcf.Total_Fit(gv.E_plot, band_centres, data3, fit_array, False,
#                     mu1_guess, mu2_guess, variance_guess, 0.1*n1, 0.1*n2)
# q4 = spcf.Total_Fit(gv.E_plot, band_centres, data4, fit_array, False,
#                     mu1_guess, mu2_guess, variance_guess, 0.1*n1, 0.1*n2)
#
# # True values from simulated observations
# mu1, sg1 = q1[0], q1[3]
# mu2, sg2 = q2[0], q2[3]
# mu3, sg3 = q3[0], q3[3]
# mu4, sg4 = q4[0], q4[3]
mu, sg = total[0], total[3]
radial_temp = total[6]
print("radial_temp: ", radial_temp)


def cost_func(t_perp_guess, t_par_guess):
    # args = collections.namedtuple('args', 'par perp comment load')

    comment = "Minimisation_Test_"
    # get estimated values
    load = False
    estimated_data = SPC.main(t_perp_guess, t_par_guess, comment, load)
    # quad1, quad2, quad3, quad4 = estimated_data[0], estimated_data[1], estimated_data[2], estimated_data[3]
    tf = estimated_data[4]
    mu_estimate, sg_estimate = tf[0], tf[3]
    # mu1_estimate, sg1_estimate = quad1[0], np.abs(quad1[3])
    # mu2_estimate, sg2_estimate = quad2[0], np.abs(quad2[3])
    # mu3_estimate, sg3_estimate = quad3[0], np.abs(quad3[3])
    # mu4_estimate, sg4_estimate = quad4[0], np.abs(quad4[3])

    # calculate cost
    mu_cost = (mu - mu_estimate)**2
    # mu_cost = (mu1 - mu1_estimate)**2 \
    # + (mu2 - mu2_estimate)**2 + (mu3 - mu3_estimate)**2 + (mu4 - mu4_estimate)**2
    sg_cost = (sg - sg_estimate)**2
    # sg_cost = (sg1 - sg1_estimate)**2 \
     # + (sg2 - sg2_estimate)**2 + (sg3 - sg3_estimate)**2 + (sg4 - sg4_estimate)**2

    cost = 0.01*mu_cost + sg_cost
    return cost


def cost_func_wrapper(args):
    t_perp = args[0]
    t_para = args[1]
    try:
        return cost_func(t_perp, t_para)
    except RuntimeError:
        return 0


def heatmap():
    temps = np.linspace(1e5, 4e5, 10).tolist()
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
    fig = plt.figure("cost function map")
    plt.figure("decoy")
    mappable = fig.gca().imshow(F, extent=[min(temps), max(temps), min(temps), max(temps)], cmap=cmap, norm=matplotlib.colors.LogNorm(np.amin(F), np.amax(F)))
    fig.colorbar(mappable, label="Cost function")
    fig.gca().set_xlabel(r"$T_{\rm{par}}$")
    fig.gca().set_ylabel(r"$T_{\rm{perp}}$")
    fig.gca().legend(title="B field at (%.02f$^\\circ$, %.02f$^\\circ$)"
               "\n from SPC normal"
               "\n True $T_{\perp}$ at 2.4e5 K"
               "\n True $T_{\parallel}$ at 1.7e5 K"
               % (np.degrees(np.arctan(gv.B[0]/gv.B[2]))
               ,  np.degrees(np.arctan(gv.B[1]/gv.B[2]))))
               # loc='center left', bbox_to_anchor=(1, 0.5)))
    fig.savefig("Corresponding_SPAN_heatmap.png")


def cost_func_1D():
    temperatures = np.linspace(1e5, 4e5, 16)
    M = []
    T_list = []
    with mp.Pool() as pool:
        t_perp_list = temperatures + 100000
        # t_par_list = temperatures - 100000
        # for i, c in enumerate(pool.imap(cost_func_wrapper, zip(temperatures, t_par_list))):
        for i, c in enumerate(pool.imap(cost_func_wrapper, zip(t_perp_list, temperatures))):
            print("\033[92mCompleted %d of %d\033[0m" % (i + 1, len(temperatures)))
            if c == 0:
                print("\033[91mFailed to converge sensibly for %d\033[0m" % (i + 1))
            else:
                M.append(c)
                T_list.append(temperatures[i])
    fig, ax = plt.subplots()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
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
    fig3.savefig("New_Par_1D_cost_func.png", bbox_inches="tight")


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
    fig2.gca().set_ylabel("Reconstructed Temperature")
    fig3.gca().set_ylabel("Reconstructed Temperature")
    fig2.gca().legend()
    fig3.gca().legend()
    fig2.savefig("perp_temp_recon.png", bbox_inches="tight")
    fig3.savefig("par_temp_recon.png", bbox_inches="tight")
    return(T_new)


grad_descent(np.array([1e9, 1e8]), radial_temp)
# heatmap()
# cost_func_1D()
