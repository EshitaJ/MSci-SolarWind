import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set()

F = np.genfromtxt('_Big-deflection_F.csv')
# print(np.min(f), np.argmin(f))
cmap = plt.cm.get_cmap('Blues', 2**16)
tck = np.linspace(1e5, 4e5, 12)
fig =plt.figure()
plt.plot(170000, 240000, 'ro', label='Expected Minimum')
mappable = fig.gca().imshow(F, extent=[min(tck), max(tck),
                                       min(tck), max(tck)],
                            cmap=cmap, norm=matplotlib.colors.LogNorm(np.amin(F), np.amax(F)))
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
