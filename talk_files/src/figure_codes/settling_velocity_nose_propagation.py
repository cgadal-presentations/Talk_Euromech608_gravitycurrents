import os
import sys
import numpy as np
import template as tp
import matplotlib.pyplot as plt
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
    DATA[fmt]['velocity'] = np.array([Position_processed[run][ind]['velocity'] for run in DATA[fmt]['runs']]) * 1e-2  # [m/s]
    DATA[fmt]['H0'] = H0
    DATA[fmt]['rho_m'] = np.array([Parameters[run]['Current density']*1e3 for run in DATA[fmt]['runs']])
    DATA[fmt]['gprime'] = (DATA[fmt]['rho_m'] - rho_f) * g / rho_f
    DATA[fmt]['vscale'] = unp.sqrt(DATA[fmt]['gprime']*DATA[fmt]['H0'])
    DATA[fmt]['REYNOLDS'] = Reynolds(DATA[fmt]['vscale'], H0, DATA[fmt]['rho_m'], mu)
    DATA[fmt]['Settling number'] = DATA[fmt]['settling_velocity']*1e-2/DATA[fmt]['vscale']
    DATA[fmt]['position'] = [Position_processed[run][ind]['position'] for run in DATA[fmt]['runs']]
    DATA[fmt]['time'] = [Position_processed[run][ind]['time'] for run in DATA[fmt]['runs']]
    DATA[fmt]['t0'] = [Position_processed[run][ind]['virtual_time_origin'] for run in DATA[fmt]['runs']]
    DATA[fmt]['x0'] = [Position_processed[run][ind]['virtual_x_origin'] for run in DATA[fmt]['runs']]
    DATA[fmt]['time_short'] = [Position_processed[run][ind]['time_short'] for run in DATA[fmt]['runs']]
    DATA[fmt]['velocity_ts'] = [Position_processed[run][ind]['velocity_ts'] for run in DATA[fmt]['runs']]
    #
    DATA[fmt]['timings'] = np.array([Position_processed[run][ind]['times_fit']
                                     for run in DATA[fmt]['runs']])[:, 1]  # [s]
    DATA[fmt]['timings'] = unp.uarray(DATA[fmt]['timings'], 0.15*DATA[fmt]['timings'])  # adding +/-1sec uncertainty to end time meas.
    #
    mask_time = DATA[fmt]['timings'] > 10.5*L_reservoir/DATA[fmt]['vscale']
    mask_vel = (DATA[fmt]['velocity'] > 0.3*DATA[fmt]['vscale']) & (DATA[fmt]['velocity'] < 0.5*DATA[fmt]['vscale'])
    DATA[fmt]['mask_ok'] = (mask_vel & mask_time)
    #
    DATA[fmt]['i_start_end'] = [Position_processed[run][ind]['indexes_fit'] for run in DATA[fmt]['runs']]
    DATA[fmt]['velocity_fit'] = [Position_processed[run][ind]['velocity_fit'] for run in DATA[fmt]['runs']]
    DATA[fmt]['virtual_x_origin'] = [Position_processed[run][ind]['virtual_x_origin'] for run in DATA[fmt]['runs']]


#%% Figure

fig, ax = plt.subplots(1, 1, layout='constrained',
                          figsize=(0.5*Beamer.fig_width, 0.8*Beamer.fig_height))

fmts = ['silibeads40_70m', 'sand80m_H19','Silibeads100_200','Silibeads150_250', 'silibeads200m_300m']

for j, fmt in enumerate(fmts):
    if fmt == 'sand80m_H19':
        run = 'run01'
    elif fmt == 'silibeads200m_300m':
        run == 'run02'
    elif fmt == 'Silibeads150_250':
        run == 'run04'
    else:
        run = 'run03'
    i = DATA[fmt]['runs'].index(run)
    S = DATA[fmt]['Settling number'][i]
    #
    if fmt in ['Silibeads150_250', 'Saline']:
        ax.plot(DATA[fmt]['time'][i] + 0.6, DATA[fmt]['position'][i]-2,
                # ax.plot(DATA[fmt]['time'][i], DATA[fmt]['position'][i],
                label=r'${:.2f}$'.format(S.n) if not np.isnan(S.n) else 'Saline',
                alpha=1, zorder=-1-j, color=tp.colors[fmt])
    else:
        ax.plot(DATA[fmt]['time'][i], DATA[fmt]['position'][i],
                label=r'${:.2f}$'.format(S.n) if not np.isnan(S.n) else 'Saline',
                alpha=1, zorder=-1-j, color=tp.colors[fmt])

ax.set_xlabel(r'Time, $t~\textrm{[s]}$')
ax.set_ylabel(r'Nose position, $x~\textrm{[cm]}$')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.legend(title=r'$S$')

fig_dir = '../figures'
fig.savefig('../figures/{}.svg'.format(sys.argv[0].split(os.sep)[-1].replace('.py', '')), dpi=600)