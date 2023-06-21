import os

import cmocean as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PyThemes.Beamer_169 as Beamer
from uncertainties import unumpy as unp

plt.style.use('./quarto.mplstyle')

# Paths to adjust before starting scripts
results_dir = '/media/cyril/LaCie_orang/petit_canal/round_spring2022/Processing/Results'
ref = 'sand80m_H19/'    # directoy to process

# #### Loading data
# Loading processed nose positions
Position_processed = np.load(os.path.join(results_dir, ref, 'nose_position/Position_processed.npy'),
                             allow_pickle=True).item()
# Loading initial parameters
Parameters = np.load(os.path.join(results_dir, ref, 'Initial_parameters.npy'),
                     allow_pickle=True).item()

# #### organizing data
runs = sorted(Position_processed.keys())
Volume_fraction = np.array([Parameters[run]['Volume_fraction'] for run in runs])
runs_sorted, phi_sorted = np.array([[run, phi] for phi, run in
                                    sorted(zip(Volume_fraction, runs))]).T
ind = -1
runs_sorted, phi_sorted = runs_sorted[1:], phi_sorted[1:]
# #### figures parameters
cmap = cmo.cm.haline_r
log_phi = np.log10(unp.nominal_values(phi_sorted))
colors = cmap((log_phi - log_phi.min())/(log_phi.max() - log_phi.min()))
colors[:, -1] = 0.7
#

exemples = ['run03', 'run11']
# exemples = ['run06']
# #### Figure
mark = 0
for i in range(3):
    fig, ax = plt.subplots(1, 1, constrained_layout=True,
                           figsize=(0.8*Beamer.fig_width, 0.95*Beamer.fig_height))
    if i == 2:
        mark = 1
    for (run, color) in zip(runs_sorted, colors):
        if mark | (run in exemples):
            index = runs.index(run)
            time = Position_processed[run][ind]['time']
            position = Position_processed[run][ind]['position']
            ax.plot(time, position, color=color)
            if i > 0:
                p = np.array([Position_processed[run][ind]['velocity_fit'].n,
                              Position_processed[run][ind]['virtual_x_origin'].n])
                istart, iend = Position_processed[run][ind]['indexes_fit']
                if run == 'run06':
                    istart, iend = Position_processed[run][-1]['indexes_fit']
                ax.plot(time[istart:iend], np.poly1d(p)(time[istart:iend]), ls='--', lw=1, color='w')

    if i == 1:
        ax.annotate(r'slope $\equiv~u_{\rm c}$', xy=(8, 50), xytext=(12, 50),
                    arrowprops=dict(arrowstyle="->", shrinkA=5, shrinkB=5))

    ax.set_xlabel('Time, $t$ (s)')
    ax.set_ylabel(r'Nose position, $x$ (cm)')
    ax.set_xlim(left=0, right=22.5)
    ax.set_ylim(bottom=0)

    norm = mpl.colors.LogNorm(vmin=phi_sorted.min().n*100, vmax=phi_sorted.max().n*100)
    phi_cmap = np.linspace(log_phi.min(), log_phi.max(), 100)
    colors_set = cmap((phi_cmap - phi_cmap.min())/(phi_cmap.max() - phi_cmap.min()))
    colors_set[:, -1] = 1
    my_cmap = mpl.colors.ListedColormap(colors_set)
    #
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap),
                 ax=ax, location='top', label=r'Volume fraction, $\phi$ [$\%$]',
                 aspect=23)

    fig.savefig(os.path.join('../figures', 'nose_positions_{}.svg'.format(i)), dpi=200)
