import os
import sys
import numpy as np
import template as tp
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from uncertainties import ufloat
import PyThemes.Beamer_169 as Beamer
from uncertainties import unumpy as unp

plt.style.use('./quarto.mplstyle')


def Reynolds(U, h, rho, mu):
    return rho*U*h/mu


# Paths to adjust before starting scripts
path_gen = '/media/cyril/LaCie_orang/petit_canal'
DATA = {}
dirs = [
    'round_spring2022/Processing/Results/sand80m_H19/',
    'round_spring2022/Processing/Results/silibeads40_70m/',
    # 'round_spring2022/Processing/Results/silibeads120m/',
    'round_spring2022/Processing/Results/silibeads200m_300m/',
    # 'round_winter2021/Processing/Results/Slope1/',
    # 'round_winter2021/Processing/Results/Slope3/',
    # 'round_winter2021/Processing/Results/Slope5/',
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
    #
    # ######## creating variable dictionnary
    fmt = dir.split(os.sep)[-2]
    DATA[fmt] = {}
    DATA[fmt]['Position_processed'] = Position_processed
    DATA[fmt]['Parameters'] = Parameters
    #
    DATA[fmt]['runs'] = sorted(Position_processed.keys())
    if fmt == 'Silibeads40_70':
        DATA[fmt]['runs'].remove('run08')  # a lot of bubbles
    DATA[fmt]['Volume_fraction'] = np.array([Parameters[run]['Volume_fraction']
                                             for run in DATA[fmt]['runs']])
    # DATA[fmt]['Stokes_velocity'] = np.array([Parameters[run]['stokes_velocity']
    #                                          for run in DATA[fmt]['runs']])
    # DATA[fmt]['Gen_settling_vel'] = np.array([Parameters[run]['Gen_Settling_velocity']
    #                                          for run in DATA[fmt]['runs']])
    DATA[fmt]['settling_velocity'] = np.array([Parameters[run]['settling_velocity']
                                               for run in DATA[fmt]['runs']])
    #
    Vreservoir = np.array([Parameters[run]['V_reservoir'] for run
                           in DATA[fmt]['runs']]) * 1e-6/W_reservoir  # [m2]
    H0 = (Vreservoir/L_reservoir)  # [m]
    #
    DATA[fmt]['velocity'] = np.array(
        [Position_processed[run][ind]['velocity'] for run in DATA[fmt]['runs']]) * 1e-2  # [m/s]
    DATA[fmt]['H0'] = H0
    DATA[fmt]['rho_m'] = np.array(
        [Parameters[run]['Current density']*1e3 for run in DATA[fmt]['runs']])
    DATA[fmt]['gprime'] = (DATA[fmt]['rho_m'] - rho_f) * g / rho_f
    DATA[fmt]['vscale'] = unp.sqrt(DATA[fmt]['gprime']*DATA[fmt]['H0'])
    DATA[fmt]['REYNOLDS'] = Reynolds(
        DATA[fmt]['vscale'], H0, DATA[fmt]['rho_m'], mu)
    #
    DATA[fmt]['position'] = [Position_processed[run][ind]['position']
                             for run in DATA[fmt]['runs']]
    DATA[fmt]['time'] = [Position_processed[run][ind]['time']
                         for run in DATA[fmt]['runs']]
    DATA[fmt]['t0'] = [Position_processed[run][ind]['virtual_time_origin']
                       for run in DATA[fmt]['runs']]
    DATA[fmt]['x0'] = [Position_processed[run][ind]['virtual_x_origin']
                       for run in DATA[fmt]['runs']]
    DATA[fmt]['time_short'] = [Position_processed[run][ind]['time_short']
                               for run in DATA[fmt]['runs']]
    DATA[fmt]['velocity_ts'] = [Position_processed[run][ind]['velocity_ts']
                                for run in DATA[fmt]['runs']]
    #
    DATA[fmt]['timings'] = np.array([Position_processed[run][ind]['times_fit']
                                     for run in DATA[fmt]['runs']])[:, 1]  # [s]
    # adding +/-1sec uncertainty to end time meas.
    DATA[fmt]['timings'] = unp.uarray(
        DATA[fmt]['timings'], 0.15*DATA[fmt]['timings'])
    #
    mask_time = DATA[fmt]['timings'] > 10.5*L_reservoir/DATA[fmt]['vscale']
    mask_vel = (DATA[fmt]['velocity'] > 0.3*DATA[fmt]['vscale']
                ) & (DATA[fmt]['velocity'] < 0.5*DATA[fmt]['vscale'])
    DATA[fmt]['mask_ok'] = (mask_vel & mask_time)
    #
    DATA[fmt]['i_start_end'] = [Position_processed[run][ind]['indexes_fit']
                                for run in DATA[fmt]['runs']]
    DATA[fmt]['velocity_fit'] = [Position_processed[run][ind]['velocity_fit']
                                 for run in DATA[fmt]['runs']]
    DATA[fmt]['virtual_x_origin'] = [Position_processed[run]
                                     [ind]['virtual_x_origin'] for run in DATA[fmt]['runs']]


for ifig in range(9):
    fig, ax = plt.subplots(1, 1, sharex=True,
                           figsize=(Beamer.fig_width, 0.93*Beamer.fig_height), layout='constrained')
    ax.set_position(Bbox([[0.12219661301172693, 0.15252121419059672], [
                    0.7372227455454167, 0.98]]))
    #
    fmts = ['Saline', 'sand80m_H19', 'Silibeads40_70',
            'silibeads200m_300m', 'Silibeads100_200', 'Silibeads150_250']
    Vss = [DATA[fmt]['settling_velocity'][0] for fmt in fmts]
    fmts_sorted, Vss_sorted = np.array([[fmt, vs] for vs, fmt in
                                        sorted(zip(Vss, fmts))]).T

    if ifig == 0:
        fmts_sorted = fmts_sorted[:1]
    elif ifig == 1:
        fmts_sorted = fmts_sorted[:2]
    elif ifig == 2:
        fmts_sorted = fmts_sorted[:3]
    #
    for fmt, vs in zip(fmts_sorted, Vss_sorted):
        tscale = L_reservoir/DATA[fmt]['vscale']
        tsed = L_reservoir/(DATA[fmt]['settling_velocity']*1e-2)
        tend = DATA[fmt]['timings']
        if ifig <= 3:
            x = unp.nominal_values(DATA[fmt]['REYNOLDS'])
            xerr = unp.std_devs(DATA[fmt]['REYNOLDS'])
            y = unp.nominal_values(tend/tscale)
            yerr = unp.std_devs(tend/tscale)
        else:
            x = unp.nominal_values(tscale/tsed)
            xerr = unp.std_devs(tscale/tsed)/2
            y = unp.nominal_values(tend/tscale)
            yerr = unp.std_devs(tend/tscale)/2
            #
        mask = DATA[fmt]['mask_ok']
        #
        a, _, _ = ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=yerr[mask],
                              fmt='.',
                              label=r'${:.2L}$'.format(
                                  vs) if not np.isnan(vs.n) else 'Saline',
                              color=tp.colors[fmt])
        ax.set_xscale('log')
        ax.set_yscale('log')

    leg1 = ax.legend(title=r'$v_{\rm s}$ [cm/s]', bbox_to_anchor=(1.02, 1),
                     loc='upper left', borderaxespad=0.)
    if (ifig <= 3):
        xth = np.logspace(-2, 0.3, 100)
        a = ax.axhline(30, color='w', ls='--', lw=1, zorder=-10,
                       label=r"$\tau = 30$")
        handles = [a]
    elif ifig >= 4:
        xth = np.logspace(-3, -1, 100)
        a = ax.axhline(30, color='w', ls='--', lw=1, zorder=-10,
                       label=r"$\tau = 30$")
        handles = [a]
        if ifig >= 5:
            b = ax.axvline(1/15, color='w', ls=':', lw=1, zorder=-10,
                           label=r'$S/a = 0.07$'
                           )
            handles = [a, b]
            #
            ax.text(0.089, 37.5, 'no \n constant velocity \n regime', rotation=60,
                    ha='center', va='center')

        if ifig >= 7:
            c,  = ax.plot(xth, 0.8*xth**-1, color='w', ls='-.', lw=1, zorder=-10,
                          label=r'$\tau = 0.8(\mathcal{S}/a)^{-1}$')
            handles = [a, b, c]

        if ifig >= 6:
            ax.axvspan(ax.get_xlim()[0], 0.013, zorder=-10, alpha=0.2,
                       color='tab:blue', linewidth=0)
            ax.text(0.0065, 12.5, 'negligible settling \n (saline currents limit)',
                    ha='center', va='center')
        if ifig >= 7:
            ax.axvspan(0.0374, 0.0667, zorder=-10, alpha=0.2,
                       color='tab:orange', linewidth=0)
            ax.text(0.05, 40, 'settling \n dominated', ha='center', va='center',
                    rotation=45)
        if ifig >= 8:
            ax.axvspan(0.013, 0.0374, zorder=-10, alpha=0.2,
                       color='tab:green', linewidth=0)
            ax.text(0.02, 12.5, 'transition', ha='center', va='center')

    if ifig <= 3:
        ax.set_ylim((8, 55))
        ax.set_xlim((2e4, 4.2e5))
        ax.set_yticks([10])
        ax.set_xlabel(r'Reynolds number, $\mathcal{R}_{e} = u_{0} h_{0}/\nu$')
        ax.set_ylabel(
            r"Dimensionless duration, $\tau = t_{\rm end}/t_{0}$")
    else:
        ax.set_xlim((0.0033, 0.12))
        ax.set_ylim((8, 55))
        ax.set_yticks([10])
        ax.set_xlabel(
            r"$\mathcal{S}/a = (v_{\rm s}/u_{0})(L_{0}/h_{0}) \sim$ Settling number")
        ax.set_ylabel(
            r"Dimensionless duration, $\tau = t_{\rm end}/t_{0}$")

    leg2 = ax.legend(handles=handles, bbox_to_anchor=(1.02, 0),
                     loc='lower left', borderaxespad=0.)
    ax.add_artist(leg1)

    fig_dir = '../figures'
    fig.savefig('../figures/{}_{}.svg'.format(
        sys.argv[0].split(os.sep)[-1].replace('.py', ''), ifig), dpi=600)
