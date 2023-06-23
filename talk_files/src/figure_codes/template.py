import numpy as np
import cmocean as cmo
import cmasher as cma
from cycler import cycler
import matplotlib.pyplot as plt
from scipy.constants import golden

# %%
# Default parameters (plt.rcParams)
# -------------------------------------
plt.style.use('./quarto.mplstyle')

# reset plt to default
# rcParams.update(rcParamsDefault)  # reset

# constants
inches_per_cm = 0.3937
regular_aspect_ratio = 1/golden

########################################
#   Figure size
########################################

fig_width = 5.84
fig_size = np.array([1, regular_aspect_ratio])*fig_width

plt.rcParams['figure.figsize'] = fig_size

# ########################################
# #   colors
# ########################################
color_list = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200',
              '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']  # colorblind friendly colorlist
plt.rcParams['axes.prop_cycle'] = cycler('color', color_list)

cmap_phi = cmo.cm.haline_r
cmap_slope = cmo.cm.ice
cmap_slope2 = cmo.cm.algae
# cmap_slope = cma.ocean

colors = {
    # #### slope
    'Sand120m_Theta0': cmap_slope(0.9),
    'Slope1': cmap_slope(0.8),
    'Slope3': cmap_slope(0.65),
    'Slope5': cmap_slope(0.45),
    'sand80m_H19': cmap_slope(0.3),
    # 'sand80m_H19': 'tab:cyan',
    'Theta7': cmap_slope2(0.8),
    'Theta10': cmap_slope2(0.5),
    'Theta15': cmap_slope2(0.3),
    # #### settling velocity
    'Saline': color_list[5],
    'Silibeads40_70': color_list[1],
    'silibeads40_70m': color_list[1],
    'Silibeads100_200': color_list[2],
    'Silibeads150_250': color_list[-2],
    # 'silibeads200m_300m': color_list[7]
    'silibeads200m_300m': 'tab:purple'
}

# # ####

# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['font.family'] = 'sans-serif'
# # plt.rcParams['font.sans-serif'] = 'StixGeneral'
# plt.rcParams['font.sans-serif'] = 'Roboto'

# plt.rcParams['font.size'] = 10  # default 12

# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['ytick.right'] = 'True'
# plt.rcParams['xtick.top'] = 'True'

# plt.rcParams['xtick.major.size'] = 2.5  # default is 3.5
# plt.rcParams['xtick.major.width'] = 0.8  # default is 0.8
# plt.rcParams['ytick.major.size'] = 2.5  # default is 3.5
# plt.rcParams['ytick.major.width'] = 0.8  # default is 0.8
