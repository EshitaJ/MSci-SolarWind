import numpy as np
import SPC
import SPC_Fits as spcf
import Global_Variables as gv
import collections
from Global_Variables import constants, load, comment

band_centres = np.genfromtxt("./Data/Band_Centres_%s_%s_%s.csv"
                             % ("Energy" if gv.E_plot else "Velocity",
                                "N_%g" % gv.N, "Field_%s" % gv.Rot))

data1 = np.genfromtxt('%s_quad_1.csv' % gv.filename)
data2 = np.genfromtxt('%s_quad_2.csv' % gv.filename)
data3 = np.genfromtxt('%s_quad_3.csv' % gv.filename)
data4 = np.genfromtxt('%s_quad_4.csv' % gv.filename)
total_data = data1 + data2 + data3 + data4

fit_array = np.linspace(np.min(band_centres), np.max(band_centres), gv.N)


mu1_guess = gv.v_sw[2] / 1e3
mu2_guess = gv.beam_v[2] / 1e3
variance_guess = 40
n1 = 0.003
n2 = 0.001

q1 = spcf.Total_Fit(gv.E_plot, band_centres, data1, fit_array, False,
                    mu1_guess, mu2_guess, variance_guess, n1, n2)
q2 = spcf.Total_Fit(gv.E_plot, band_centres, data2, fit_array, False,
                    mu1_guess, mu2_guess, variance_guess, n1, n2)
q3 = spcf.Total_Fit(gv.E_plot, band_centres, data3, fit_array, False,
                    mu1_guess, mu2_guess, variance_guess, n1, n2)
q4 = spcf.Total_Fit(gv.E_plot, band_centres, data4, fit_array, False,
                    mu1_guess, mu2_guess, variance_guess, n1, n2)
total = spcf.Total_Fit(gv.E_plot, band_centres, total_data, fit_array, True,
                       mu1_guess, mu2_guess, variance_guess, 0.09, 0.01)

# True values from simulated observations
mu1, sg1 = q1[0], q1[3]
mu2, sg2 = q2[0], q2[3]
mu3, sg3 = q3[0], q3[3]
mu4, sg4 = q4[0], q4[3]
radial_temp = total[6]
print("radial_temp: ", radial_temp)


def cost_func(t_perp_guess, t_par_guess):
    # args = collections.namedtuple('args', 'par perp comment load')

    constants["T_perp"] = t_perp_guess
    constants["T_par"] = t_par_guess
    comment = "Minimisation_Test_"

    # get estimated values
    load = False
    # gv.args = args
    estimated_data = SPC.main()
    quad1, quad2, quad3, quad4 = estimated_data
    mu1_estimate, sg1_estimate = quad1[0], np.abs(quad1[3])
    mu2_estimate, sg2_estimate = quad2[0], np.abs(quad2[3])
    mu3_estimate, sg3_estimate = quad3[0], np.abs(quad3[3])
    mu4_estimate, sg4_estimate = quad4[0], np.abs(quad4[3])

    # calculate cost
    mu_cost = (mu1 - mu1_estimate)**2 + (mu2 - mu2_estimate)**2 \
        + (mu3 - mu3_estimate)**2 + (mu4 - mu4_estimate)**2
    sg_cost = (sg1 - sg1_estimate)**2 + (sg2 - sg2_estimate)**2 \
        + (sg3 - sg3_estimate)**2 + (sg4 - sg4_estimate)**2

    cost = 0.01*mu_cost + sg_cost
    return cost


def grad_descent(coeff, t_estimate):

    def grad_cost(delta, t):
        t_perp_est, t_par_est = t
        delta_t_perp = t_perp_est + delta*t_perp_est
        delta_t_par = t_par_est + delta*t_par_est

        grad_cost_perp = (cost_func(delta_t_perp,
                                    t_par_est)
                          - cost_func(t_perp_est,
                                      t_par_est)) / (delta*t_perp_est)
        grad_cost_par = (cost_func(t_perp_est,
                                   delta_t_par)
                         - cost_func(t_perp_est,
                                     t_par_est)) / (delta*t_par_est)
        grad = np.array([grad_cost_perp, grad_cost_par])
        print("grad: ", grad)
        return grad

    T_0 = np.array([t_estimate, t_estimate])  # perp, par
    T_old = T_0
    T_new = T_old
    error = np.inf
    iteration = 0
    epsilon = 0.01

    while error > epsilon:
        T_new, T_old = T_old - coeff*grad_cost(0.01, T_old), T_new
        error = np.linalg.norm((T_new - T_old)**2) / np.linalg.norm((T_old)**2)
        print("iteration: ", iteration)
        print("perp, par: ", T_new)
        iteration += 1

    print("final perp, par: ", T_new)
    return(T_new)


grad_descent(1e3, radial_temp)
