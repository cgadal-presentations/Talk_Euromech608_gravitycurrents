import os
import sys

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import PyThemes.Beamer_169 as Beamer
import template as tp
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Ellipse
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from uncertainties import ufloat
from uncertainties import unumpy as unp

plt.style.use('./quarto.mplstyle')


class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Ellipse(xy=center, width=width + xdescent,
                    height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def E_Cenedese(Fr, Re0, alpha=7.18, F0=0.51, A=3.4*1e-3, B=243.52,
               Min=4*1e-5, Max=1, Rec=6*1e4):
    Re = Re0*0.4*0.4  # uc*h/nu
    Cinf = (1/Max) + (Rec/Re)**(1/2)
    E = (Min + A*Fr**alpha)/(1 + A*Cinf*(Fr + F0)**alpha)
    return E


def compute_arc_length(x, y):
    return np.trapz(np.sqrt(1 + np.gradient(y, x) ** 2), x)


def Reynolds(U, h, rho, mu):
    return rho*U*h/mu


data_Balasubramanian2018 = np.array([
    [5495.002537016734, 0.011954545454545454],
    [484.61098954332084, 0.0007272727272727283],
    [1507.1974284408825, 0.004022727272727273],
    [2071.151549145493, 0.003454545454545455],
    [2732.7324303201617, 0.004409090909090909],
    [3087.2131337590504, 0.004295454545454546],
    [4258.076730412151, 0.007954545454545454],
    [8070.530565924998, 0.012772727272727272],
    [10148.936746066307, 0.015045454545454542],
    [12299.505200127782, 0.019204545454545453],
    [985.3248916814835, 0.00215909090909091],
])

#
data_ottolenghi_tp = np.loadtxt('src/data_ottolenghi2016.csv',
                                skiprows=2, usecols=(1, 2, 3, 4, 5, 6, 7, 8),
                                delimiter=',')
mask = data_ottolenghi_tp[:, -1] < 10
data_ottolenghi = {}
H0 = data_ottolenghi_tp[:, 1]*data_ottolenghi_tp[:, 2]
nu = (data_ottolenghi_tp[:, -5]*H0/data_ottolenghi_tp[:, -4])
Re = np.sqrt(data_ottolenghi_tp[:, 0]*H0)*H0/nu
data_ottolenghi['Re_0'] = Re[mask]
data_ottolenghi['E'] = data_ottolenghi_tp[:, -1][mask]
#
data_nogueira_tp = np.loadtxt('src/nogueira2014_values.csv',
                              skiprows=2, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                   10, 11, 12),
                              delimiter=';')
data_nogueira = {}
H0 = 0.2
data_nogueira['Re_0'] = H0 * np.sqrt(data_nogueira_tp[:-3, 4]*H0)/nu.mean()
data_nogueira['E'] = data_nogueira_tp[:-3, -1]
#
data_jacobson_tp = np.loadtxt('src/Jacobson2014_values.csv',
                              skiprows=2, delimiter=',')
gprime = 9.81*(data_jacobson_tp[:, 3] - 0.998)/0.998
u0 = np.sqrt(gprime*data_jacobson_tp[:, 4]*1e-2)
data_jacobson = {}
data_jacobson['Re_0'] = u0*data_jacobson_tp[:, 4]*1e-2/nu.mean()
data_jacobson['E'] = data_jacobson_tp[:, -1]


# Paths to adjust before starting scripts
path_gen = '/media/cyril/LaCie_orang/petit_canal'
DATA = {}
dirs = [
    'round_spring2022/Processing/Results/sand80m_H19/',
    # 'round_spring2022/Processing/Results/silibeads40_70m/',
    # 'round_spring2022/Processing/Results/silibeads120m/',
    'round_spring2022/Processing/Results/silibeads200m_300m/',
    'round_winter2021/Processing/Results/Slope1/',
    'round_winter2021/Processing/Results/Slope3/',
    'round_winter2021/Processing/Results/Slope5/',
    # 'round_winter2021/Processing/Results/Slope7/',
    'round_winter2022/Processing/Results/Sand120m_Theta0/',
    #
    'round_winter2022/Processing/Results/Silibeads40_70/',
    'round_winter2022/Processing/Results/Silibeads100_200/',
    'round_winter2022/Processing/Results/Silibeads150_250/',
    'round_winter2022/Processing/Results/Saline/',
]

# #### parameters
ind = -1
g = 9.81  # [m/s2]
rho_f = ufloat(0.998, 0.001)*1e3  # [kg/m3]
rho_p = ufloat(2.65, 0.02)*1e3  # [kg/m3]
mu = ufloat(1, 0.1)*10**-3  # water dynamic viscosity [kg/(m·s)]
L_reservoir = ufloat(9.9, 0.2)*1e-2  # [m]
W_reservoir = ufloat(19.4, 0.2)*1e-2  # [m]


for dir in dirs:
    path_total = os.path.join(path_gen, dir)
    # Loading processed nose positions
    Position_processed = np.load(os.path.join(path_total, 'nose_position/Position_processed.npy'),
                                 allow_pickle=True).item()
    # Loading initial parameters
    Parameters = np.load(os.path.join(path_total, 'Initial_parameters.npy'),
                         allow_pickle=True).item()
    # Loading shape properties
    SHAPES_props = np.load(os.path.join(path_total, 'shape/av_shapes_log/Shape_logs_props.npy'),
                           allow_pickle=True).item()
    # # Loading shape properties
    SHAPES_props_time = np.load(os.path.join(path_total, 'shape/shape_time/shape_props_time.npy'),
                                allow_pickle=True).item()
    # Loading shapes
    SHAPES = np.load(os.path.join(
        path_total, 'shape/av_shapes/Av_shapes.npy'), allow_pickle=True).item()
    #
    door_opening = np.load(os.path.join(path_total, 'door_opening', 'door_times.npy'),
                           allow_pickle=True).item()
    # ######## creating variable dictionnary
    fmt = dir.split(os.sep)[-2]
    DATA[fmt] = {}
    DATA[fmt]['runs'] = sorted(Position_processed.keys())
    DATA[fmt]['Volume_fraction'] = np.array([Parameters[run]['Volume_fraction']
                                             for run in DATA[fmt]['runs']])
    #
    DATA[fmt]['slope'] = Parameters[DATA[fmt]['runs'][0]]['slope']
    Vreservoir = np.array([Parameters[run]['V_reservoir']
                          for run in DATA[fmt]['runs']]) * 1e-6/W_reservoir  # [m2]
    H0 = (Vreservoir/L_reservoir)  # [m]
    # #### Entrainment from average shape
    Vcourant_n = np.array([np.trapz(unp.nominal_values(SHAPES[run]['shape']), SHAPES[run]['xcenters'])
                           for run in DATA[fmt]['runs']]) * 1e-4  # [m2]
    Vcourant_min = np.array([np.trapz(unp.nominal_values(SHAPES[run]['shape']) - unp.std_devs(SHAPES[run]['shape']), SHAPES[run]['xcenters'])
                             for run in DATA[fmt]['runs']]) * 1e-4  # [m2]
    Vcourant = unp.uarray(Vcourant_n, np.abs(Vcourant_n - Vcourant_min))
    # #
    velocity = np.array([Position_processed[run][-1]['velocity']
                        for run in DATA[fmt]['runs']]) * 1e-2  # [m/s]
    # Tend = np.array([Position_processed[run][-1]['times_fit'][-1] for run in DATA[fmt]['runs']])
    # HB = np.array([SHAPES_props[run]['hc_benjamin'] for run in DATA[fmt]['runs']])/2 * 1e-2  # [m]
    # E = (Vcourant - Vreservoir)/Tend/velocity/HB
    # #
    # Gamma = np.array([compute_arc_length(SHAPES[run]['xcenters'],
    #                                      unp.nominal_values(SHAPES[run]['shape']))
    #                   for run in DATA[fmt]['runs']]) * 1e-2  # [m]
    # xend = np.array([Position_processed[run][-1]['position'][Position_processed[run][-1]['indexes_fit'][-1]] for run in DATA[fmt]['runs']]) * 1e-2  # [m]
    # E_area = (1/xend)*(Vcourant - Vreservoir)/xend
    # E_area_base = (Vcourant - Vreservoir)/xend/Gamma
    #
    # #### entrainment from max volume
    # max volume before it reaches the tank end
    Iend = np.array([np.argwhere(~np.isnan(Position_processed[run][-1]['position']))[-1][0]
                     for run in DATA[fmt]['runs']])
    V_courant_max = np.array([np.nanmax(gaussian_filter1d(unp.nominal_values(SHAPES_props_time[run]['Area'])[:iend], sigma=10))
                              for run, iend in zip(DATA[fmt]['runs'], Iend)]) * 1e-4  # [m2]
    V_courant_max = unp.uarray(
        V_courant_max, np.abs(Vcourant_n - Vcourant_min))
    i_Vmax = np.array([np.nanargmax(gaussian_filter1d(unp.nominal_values(SHAPES_props_time[run]['Area'])[:iend], sigma=10))
                      for run, iend in zip(DATA[fmt]['runs'], Iend)])
    Gamma_max = np.array([gaussian_filter1d(unp.nominal_values(SHAPES_props_time[run]['Gamma']), sigma=10)[i]
                          for run, i in zip(DATA[fmt]['runs'], i_Vmax)]) * 1e-2  # [m2]
    xend_max = np.array([Position_processed[run][-1]['position'][i]
                         for run, i in zip(DATA[fmt]['runs'], i_Vmax)]) * 1e-2  # [m]
    E_max = (V_courant_max - Vreservoir)/xend_max/Gamma_max

    # max volume in the slumping regime
    # Iend_lin = np.array([Position_processed[run][-1]['indexes_fit'][-1]
    #                      for run in DATA[fmt]['runs']])
    # V_courant_lin = np.array([np.nanmax(SHAPES_props_time[run]['Area'][:iend])
    #                           for run, iend in zip(DATA[fmt]['runs'], Iend_lin)]) * 1e-4  # [m2]
    # V_courant_lin = unp.uarray(V_courant_lin, np.abs(Vcourant_n - Vcourant_min))
    # i_Vlin = np.array([np.nanargmax(SHAPES_props_time[run]['Area'][:iend])
    #                   for run, iend in zip(DATA[fmt]['runs'], Iend)])
    # Gamma_lin = np.array([SHAPES_props_time[run]['Gamma'][i]
    #                       for run, i in zip(DATA[fmt]['runs'], i_Vmax)]) * 1e-2  # [m2]
    # xend_lin = np.array([Position_processed[run][-1]['position'][i]
    #                      for run, i in zip(DATA[fmt]['runs'], i_Vlin)]) * 1e-2  # [m]
    # E_lin = (V_courant_lin - Vreservoir)/xend_lin/Gamma_lin
    #
    # shape_indicator = np.array([True if (np.nanmean(SHAPES[run]['shape'][0:5]) < 5)
    # else False for run in DATA[fmt]['runs']])
    # shape_indicator = DATA[fmt]['Volume_fraction'] < 2.5*1e-2
    #
    DATA[fmt]['E'] = E_max
    DATA[fmt]['Gamma'] = Gamma_max
    DATA[fmt]['Vcourant'] = V_courant_max
    DATA[fmt]['xend'] = xend_max
    # DATA[fmt]['E'] = E_area_base
    # DATA[fmt]['Gamma'] = Gamma
    # DATA[fmt]['Vcourant'] = Vcourant
    # DATA[fmt]['E'] = E_lin
    # DATA[fmt]['E'] = E_max
    DATA[fmt]['H0'] = H0
    DATA[fmt]['Vcourant'] = Vcourant
    DATA[fmt]['Vreservoir'] = Vreservoir
    DATA[fmt]['velocity'] = velocity
    DATA[fmt]['rho_m'] = np.array(
        [Parameters[run]['Current density']*1e3 for run in DATA[fmt]['runs']])
    DATA[fmt]['gprime'] = (DATA[fmt]['rho_m'] - rho_f) * g / rho_f
    DATA[fmt]['vscale'] = unp.sqrt(DATA[fmt]['gprime']*DATA[fmt]['H0'])
    # DATA[fmt]['REYNOLDS'] = Reynolds(DATA[fmt]['vscale'], H0, rho_f, mu)
    DATA[fmt]['REYNOLDS'] = Reynolds(
        DATA[fmt]['vscale'], H0, DATA[fmt]['rho_m'], mu)
    # DATA[fmt]['REYNOLDS'] = Reynolds(DATA[fmt]['velocity'], H0, DATA[fmt]['rho_m'], mu)
    # DATA[fmt]['REYNOLDS'] = Reynolds(DATA[fmt]['velocity'], H0, DATA[fmt]['rho_m'], mu)
    # DATA[fmt]['dt_door'] = dt_door
    # shape_indicator = DATA[fmt]['dt_door'] < 10*DATA[fmt]['H0']/DATA[fmt]['vscale']
    # shape_indicator = DATA[fmt]['dt_door'] < 10*DATA[fmt]['H0']/DATA[fmt]['vscale']
    # DATA[fmt]['shape_indicator'] = shape_indicator
    #
    DATA[fmt]['timings'] = np.array([Position_processed[run][ind]['times_fit']
                                     for run in DATA[fmt]['runs']])[:, 1]  # [s]
    mask_time = DATA[fmt]['timings'] > 10.5*L_reservoir/DATA[fmt]['vscale']
    mask_vel = (DATA[fmt]['velocity'] > 0.3*DATA[fmt]['vscale']
                ) & (DATA[fmt]['velocity'] < 0.5*DATA[fmt]['vscale'])
    DATA[fmt]['mask_ok'] = (mask_vel & mask_time)
    #
    DATA[fmt]['settling_velocity'] = np.array([Parameters[run]['settling_velocity']
                                               for run in DATA[fmt]['runs']])

# ######## Figure
for ifig in range(3):
    figsize = (Beamer.fig_width, Beamer.fig_height)
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    alphas = [0.85, 0.15]
    markers = ['.', 's']
    alpha_others = 0.15

    ax.set_xscale('log')
    ax.set_yscale('log')

    if ifig > 0:
        # ####
        color_saline = 'tab:purple'
        alpha_saline = 0.7
        bala = ax.scatter(2*data_Balasubramanian2018[:, 0]/0.4, data_Balasubramanian2018[:, 1],
                          color=color_saline, alpha=alpha_saline, linewidth=0, marker='^')

        nogueira2014 = ax.scatter(data_nogueira['Re_0'], data_nogueira['E'],
                                  color=color_saline, alpha=alpha_saline, linewidth=0,
                                  marker='s')

        ottolenghi2016 = ax.scatter(data_ottolenghi['Re_0'], data_ottolenghi['E'],
                                    color=color_saline, alpha=alpha_saline, linewidth=0,
                                    marker='*', s=12**2)
        ##
        # Sher_rect = plt.Rectangle([2*3000/0.4, 0.7], 2*36000/0.4, 0.1, color='tab:orange',
        #                           alpha=0.2)
        # ax.add_patch(Sher_rect)
        #####
        color_turbid = 'tab:green'
        alpha_turbid = 0.4
        #
        Wilson_rect = plt.Rectangle([2*11500/0.4, 0.04], 2*55000/0.4, 0.02,
                                    color=color_turbid, alpha=alpha_turbid, linewidth=0)
        ax.add_patch(Wilson_rect)
        ##

        jacobson2014 = ax.scatter(data_jacobson['Re_0'], data_jacobson['E'],
                                  color=color_turbid, alpha=alpha_turbid, linewidth=0,
                                  marker='d')

    # #### plotting points without errorbars
    x = np.concatenate([DATA[fmt]['REYNOLDS'][DATA[fmt]['mask_ok']]
                        for fmt in sorted(DATA.keys())])
    y = np.concatenate([DATA[fmt]['E'][DATA[fmt]['mask_ok']]
                        for fmt in sorted(DATA.keys())])
    colors = np.concatenate([[to_rgba(tp.colors[fmt]) for i in DATA[fmt]['E'][DATA[fmt]['mask_ok']]]
                            for fmt in sorted(DATA.keys())])

    plot_idx = np.arange(x.size)
    np.random.shuffle(plot_idx)
    ax.scatter(unp.nominal_values(x)[plot_idx], unp.nominal_values(y)[plot_idx],
               alpha=1, c=colors[plot_idx[:, None]], linewidth=0)

    # #### adding some errobars for some points
    to_annotate = [
        ('run14', 'sand80m_H19'),
        ('run02', 'sand80m_H19'),
        ('run01', 'Sand120m_Theta0'),
        ('run10', 'Silibeads40_70'),
        ('run12', 'Silibeads40_70'),
        ('run01', 'Saline'),
        ('run13', 'Saline'),
        ('run05', 'Silibeads100_200'),
        ('manip12', 'Slope3'),
        ('manip02', 'Slope5'),
        # ('run01', 'Silibeads150_250'),
    ]

    for run, fmt in to_annotate:
        i = DATA[fmt]['runs'].index(run)
        x = DATA[fmt]['REYNOLDS'][i:i+1]
        y = DATA[fmt]['E'][i:i+1]
        #
        a, _, _ = ax.errorbar(unp.nominal_values(x), unp.nominal_values(y),
                              xerr=unp.std_devs(x), yerr=unp.std_devs(y), fmt='.',
                              alpha=1, color=tp.colors[fmt],
                              markeredgewidth=0, markersize=10,
                              )

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    Re_th = np.logspace(3, 6, 100)
    if ifig > 1:
        ax.plot(Re_th, Re_th/(0.2*1e7), 'w--',
                label=r'$E \propto \mathcal{R}e$')
    # ax.plot(Re_th, E_Cenedese(1.2, Re_th))
    ax.set_xlim(1894.4879332251328, 425268.3566815987)
    ax.set_ylim(0.0006, 0.07481246211278558)

    bad_points = Ellipse((0.83, 0.9), width=0.29, height=0.135, transform=ax.transAxes,
                         label='biased points', facecolor='none', edgecolor='w')
    ax.add_patch(bad_points)
    ##
    ax.set_xlabel(
        r"Reynolds number, $\mathcal{R}e = \rho_{0} \sqrt{g_{0}'H_{0}} H_{0}/\eta $")
    ax.set_ylabel(r"Entrainment coefficient, $E$")

    leg1 = ax.legend(loc='center left', handler_map={
                     Ellipse: HandlerEllipse()})
    if ifig > 0:
        leg2 = ax.legend([bala,
                          ottolenghi2016,
                          # Sher_rect,
                          nogueira2014,
                          ],
                         ['Balasubramanian et al. 2018',
                         'Ottolenghi et al. 2016',
                          # 'Sher et al. 2015',
                          'Nogueira et al. 2014',
                          ],
                         title='Saline currents', loc='lower right')

        leg3 = ax.legend([Wilson_rect, jacobson2014],
                         ['Wilson et al. 2017', 'Jacobson et al. 2014'],
                         title='Turbidity currents',
                         bbox_to_anchor=(1, 0.35),
                         loc='lower right')
        ax.add_artist(leg1)
        ax.add_artist(leg2)

    fig.align_labels()

    fig_dir = '../figures'
    fig.savefig(
        '../figures/{}_{}.svg'.format(sys.argv[0].split(os.sep)[-1].replace('.py', ''), ifig), dpi=600)
