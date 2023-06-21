import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import PyThemes.Beamer_169 as Beamer
import template as tp
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp

plt.style.use('./quarto.mplstyle')


def phi2drho(phi, rho_f=998, rho_p=2650):
    return phi*1e-2*(rho_p-rho_f)


def drho2phi(drho, rho_f=998, rho_p=2650):
    return 1e2*drho/(rho_p-rho_f)


def lin(x, b):
    return x + b


def aff(x, a, b):
    return a*x + b


def Froude(U, g, h):
    return U/unp.sqrt(g*h)


def Reynolds(U, h, rho, mu):
    return rho*U*h/mu


def Benj_correction(alpha):
    return np.sqrt(alpha*(1 - alpha)*(2 - alpha)/(1 + alpha))


def Slope_Froude_squared(theta, Fr0, c, a):
    return Fr0**2*(np.cos(np.radians(theta)) + c*a*np.sin(np.radians(theta)))


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
    'round_winter2022/Processing/Results/Silibeads40_70/',
    'round_winter2022/Processing/Results/Silibeads100_200/',
    'round_winter2022/Processing/Results/Silibeads150_250/',
    'round_winter2022/Processing/Results/Saline/',
    'round_winter2022/Processing/Results/Sand120m_Theta0/',
    ]

# DATA2 = {}
# dirs_2 = [
#     'Manips_LEGI/Processing/Results/Theta7/image_processing/',
#     'Manips_LEGI/Processing/Results/Theta10/image_processing/',
#     'Manips_LEGI/Processing/Results/Theta15/image_processing/',
# ]

# #### parameters
ind = -1
g = 9.81  # [m/s2]
rho_f = ufloat(0.998, 0.001)*1e3  # [kg/m3]
rho_p = ufloat(2.65, 0.1)*1e3  # [kg/m3]
mu = ufloat(8.90, 1)*10**-4  # water dynamic viscosity [kg/(m·s)]
L_reservoir = ufloat(9.9, 0.2)*1e-2  # [m]
W_reservoir = ufloat(19.4, 0.2)*1e-2  # [m]

for dir in dirs:
    path_total = os.path.join(path_gen, dir)
    # Loading processed nose positions
    Position_processed = np.load(os.path.join(path_total, 'nose_position/Position_processed.npy'),
                                 allow_pickle=True).item()
    # Loading raw nose positions
    nose_positions = np.load(os.path.join(path_total, 'nose_position/nose_positions.npy'),
                             allow_pickle=True).item()
    # Loading initial parameters
    Parameters = np.load(os.path.join(path_total, 'Initial_parameters.npy'),
                         allow_pickle=True).item()

    # ######## creating variable dictionnary
    fmt = dir.split(os.sep)[-2]
    DATA[fmt] = {}
    DATA[fmt]['Position_processed'] = Position_processed
    DATA[fmt]['nose_positions'] = nose_positions
    DATA[fmt]['Parameters'] = Parameters
    #
    DATA[fmt]['runs'] = sorted(Position_processed.keys())
    DATA[fmt]['slope'] = Parameters[DATA[fmt]['runs'][0]]['slope']
    if fmt == 'Silibeads40_70':
        DATA[fmt]['runs'].remove('run08')  # a lot of bubbles
    DATA[fmt]['Volume_fraction'] = np.array([Parameters[run]['Volume_fraction']
                                             for run in DATA[fmt]['runs']])
    DATA[fmt]['velocity'] = np.array([Position_processed[run][ind]['velocity']
                                      for run in DATA[fmt]['runs']])*1e-2  # [m/s]
    if 'Slope' in fmt:
        DATA[fmt]['velocity'] = DATA[fmt]['velocity']*1.007  # correction from calibration bias
    Vreservoir = np.array([Parameters[run]['V_reservoir'] for run
                           in DATA[fmt]['runs']]) * 1e-6/W_reservoir  # [m2]
    H0 = (Vreservoir/L_reservoir)  # [m]
    #
    DATA[fmt]['H0'] = H0
    # DATA[fmt]['rho_m'] = rho_f*(1 + DATA[fmt]['Volume_fraction']*(rho_p-rho_f)/rho_f)
    DATA[fmt]['rho_m'] = np.array([Parameters[run]['Current density']*1e3 for run in DATA[fmt]['runs']])
    DATA[fmt]['gprime'] = (DATA[fmt]['rho_m'] - rho_f) * g / rho_f
    DATA[fmt]['vscale'] = unp.sqrt(DATA[fmt]['gprime']*DATA[fmt]['H0'])
    DATA[fmt]['FROUDES'] = DATA[fmt]['velocity']/DATA[fmt]['vscale']
    DATA[fmt]['REYNOLDS'] = Reynolds(DATA[fmt]['vscale'], DATA[fmt]['H0'],
                                     DATA[fmt]['rho_m'], mu)
    DATA[fmt]['timings'] = np.array([Position_processed[run][ind]['times_fit']
                                     for run in DATA[fmt]['runs']])[:, 1]  # [s]
    # mask_time = DATA[fmt]['timings'] > 0
    # mask_vel = (DATA[fmt]['velocity'] > 0)
    mask_time = DATA[fmt]['timings'] > 10.5*L_reservoir/DATA[fmt]['vscale']
    mask_vel = (DATA[fmt]['velocity'] > 0.3*DATA[fmt]['vscale']) & (DATA[fmt]['velocity'] < 0.5*DATA[fmt]['vscale'])
    mask_Re = DATA[fmt]['REYNOLDS'] > 2e4
    DATA[fmt]['mask_ok'] = (mask_vel & mask_time & mask_Re)
    DATA[fmt]['set-up'] = 1
    DATA[fmt]['settling_velocity'] = np.array([Parameters[run]['settling_velocity']
                                               for run in DATA[fmt]['runs']])

save_dir = '../Figures'
# ################# U = f(gh) plots
for i in range(5):
    fig, ax = plt.subplots(1, 1, sharex=True,
                           figsize=(Beamer.fig_width, 0.93*Beamer.fig_height), layout='constrained')
    if (i == 1) | (i == 3):
        fmts_sorted = ['sand80m_H19']
    elif i == 2:
        fmts_sorted = ['Saline', 'Silibeads40_70', 'sand80m_H19']
    else:
        fmts_all = ['Sand120m_Theta0', 'Slope1', 'Slope3', 'Slope5', 'sand80m_H19']
        # fmts_sorted = ['Slope1', 'sand80m_H19']
        fmts_sorted = ['Sand120m_Theta0', 'sand80m_H19']
    if i < 1:
        Par_sorted = [DATA[fmt]['settling_velocity'][0] for fmt in fmts_sorted]
    else:
        Par_sorted = [DATA[fmt]['slope'] for fmt in fmts_sorted]
    #
    if i > 0:
        for fmt, par in zip(fmts_sorted, Par_sorted):
            Reynolds = DATA[fmt]['REYNOLDS'][DATA[fmt]['mask_ok']]
            Froude = DATA[fmt]['FROUDES'][DATA[fmt]['mask_ok']]
            #
            x = unp.nominal_values(Reynolds)
            y = unp.nominal_values(Froude)
            xerr = unp.std_devs(Reynolds)
            yerr = unp.std_devs(Froude)
            #
            label = r'${:.2L}$'.format(par) if not np.isnan(par.n) else r'$\textrm{Saline}$'
            a, _, _ = ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='.', label=label,
                                color=tp.colors[fmt])
    #     # ## fit
    #     if (i > 1) | (fmt == 'sand80m_H19'):
    #         x_fit = np.logspace(np.log10(Reynolds.min().n*0.25),
    #                             np.log10(Reynolds.max().n*1.2), 100)
    #         mask = vscale > 8
    #         A = []
    #         a = []
    #         xlog = unp.log(vscale[mask])
    #         ylog = unp.log(velocity[mask])
    #         p, pcov = curve_fit(lin, unp.nominal_values(xlog), unp.nominal_values(ylog),
    #                             sigma=unp.nominal_values(ylog))
    #         x_fit = np.logspace(np.log10(x.min()*0.9), np.log10(x.max()*1.1), 100)
    #         ax.plot(x_fit, np.exp(p[0])*x_fit, ls='--',
    #                 color=tp.colors[fmt] if i > 1 else 'w', alpha=1,
    #                 # label=r'$u_{\rm c} \propto u_{0}$' if fmt == 'sand80m_H19' else None
    #                 )
    # #
    # if i > 3:
    #     axins = ax.inset_axes([0.65, -0.26, 0.57, 0.67])
    #     Fr = [np.mean(DATA[fmt]['FROUDES']) for fmt in fmts_all]
    #     slopes = [DATA[fmt]['slope'] for fmt in fmts_all]
    #     #
    #     axins.errorbar(unp.nominal_values(slopes), unp.nominal_values(Fr),
    #                    xerr=unp.std_devs(slopes), yerr=unp.std_devs(Fr),
    #                    fmt='.')
    #     axins.set_xlabel(r'$\textrm{Slope},~\theta~[^\circ]$')
    #     axins.set_ylabel(r'$\mathcal{F}r = u_{\rm c}/u_{0}$')
    #     # axins.set_ylim(bottom=1)
    #     # axins.set_xlim(left=0)
    #
    leg1 = ax.legend(loc='lower right',
                     title=r'$v_{\rm s}~[\textrm{cm/s}]$' if i < 3 else r'$\theta~[^\circ]$',
                     ncol=2 if i > 3 else 1)
    ax.add_artist(leg1)
    #
    ax.set_xscale('log')
    ax.set_ylim([0, 0.6])
    ax.set_xlim([2e4, 4.2e5])
    # ax.set_yscale('log')
    ax.set_ylabel(r'$\mathcal{F}_{r} = u_{\rm c}/u_{0}$')
    ax.set_xlabel(r'$\mathcal{R}_{e} = u_{0} h_{0}/\nu$')


    fig_dir = '../figures'
    fig.savefig('../figures/{}_{}.svg'.format(sys.argv[0].split(os.sep)[-1].replace('.py', ''), i), dpi=600)