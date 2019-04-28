import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
# import seaborn as sns
# sns.set()

if len(sys.argv) != 4:
    print("Please specify a resolution")
    print("Usage: %s <resolution> <B field> <comment>" % tuple(sys.argv[1:]))
    sys.exit(1)

F = np.genfromtxt('%s_%s_F%s.csv' % (sys.argv[1], sys.argv[2], sys.argv[3]))
# print(np.min(f), np.argmin(f))
cmap = plt.cm.get_cmap('Blues', 2**16)

min_t = 1e5
max_t = 3e5
nsamples = int(sys.argv[1])
expected_min = (170000, 240000)

tck = np.linspace(min_t, max_t, nsamples)
fig = plt.figure()
plt.plot(*expected_min, 'ro', label='Expected Minimum')

min_arg = np.unravel_index(np.argmin(F), F.shape)
print("Minimum value in mappable: %g, %g" % min_arg)
print("True minimum temperature:  %g, %g" % expected_min)

temp_arg = (0.5 + np.array(min_arg)) * ((max_t - min_t) / nsamples) + min_t
print("Est. minimum temperature: %g, %g" % tuple(temp_arg[::-1].tolist()))


# print("Corresponding temperature: %g, %g" % ())
mappable = fig.gca().imshow(F, extent=[min(tck), max(tck),
                                       min(tck), max(tck)],
                            cmap=cmap, norm=matplotlib.colors.LogNorm(np.amin(F), np.amax(F)), origin="lower")

plt.plot(temp_arg[1], temp_arg[0], 'bx', label='Estimated Minimum \n at ( %g (K), %g (K) )' % tuple(temp_arg[::-1].tolist()))
fig.colorbar(mappable, label="Cost function")
fig.gca().set_xlabel("$T_{\parallel}$ (K)")
fig.gca().set_ylabel("$T_{\perp}$ (K)")
fig.gca().legend()
# fig.gca().legend(title="B field at (%.02f$^\\circ$, %.02f$^\\circ$)"
#            "\n from SPC normal"
#            "\n True $T_{\perp}$ at 2.4e5 K"
#            "\n True $T_{\parallel}$ at 1.7e5 K"
#            % (np.degrees(np.arctan(gv.B[0]/gv.B[2]))
#            ,  np.degrees(np.arctan(gv.B[1]/gv.B[2]))),
#            loc='center left', bbox_to_anchor=(1.5, 0.5))
plt.show()
