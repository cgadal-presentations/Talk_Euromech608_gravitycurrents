import os

import matplotlib.pyplot as plt
import numpy as np
import PyThemes.Beamer_169 as Beamer
from General.Math import Rotation_matrix, cosd, sind

fig_dir = '../figures'

# colors
color_water = 'deepskyblue'
color_sed = 'peru'
color_walls = 'k'
alpha_water = 0.15
color_mixing = 'grey'

# dimension parameters
slope = -5  # degrees
tank_height = 45  # cm
tank_length = 165  # cm
door_pos = 15  # cm
door_height = 0.9*tank_height
door_pad = 0*tank_height
water_height = 40  # cm, at end of the canal
y_bottom = 0  # cm, at the end of the canal
x_bottom = 0  # cm, at the end of the canal
mixing_height = tank_height
mixing_pad = 0.2*tank_height
mixing_width = 0.15*mixing_height

# ## reference points
slope_vec = np.array([cosd(slope), sind(slope)])
slope_vec_up = np.array([-sind(slope), cosd(slope)])
down_vec = np.array([0, -1])
# tank
bottom_right = np.array([x_bottom, y_bottom])
bottom_left = bottom_right - tank_length*slope_vec
top_right = bottom_right + tank_height*slope_vec_up
top_left = bottom_left + tank_height*slope_vec_up
# door
bottom_door = bottom_left + door_pos*slope_vec + door_pad*slope_vec_up
top_door = bottom_door + door_height*slope_vec_up
# water
bottom_left_water = bottom_left
bottom_right_water = bottom_right
top_right_water = bottom_right_water + water_height*slope_vec_up
top_left_water = top_right_water - tank_length*np.array([1, 0])/cosd(slope)
xy_water = np.array([bottom_left_water, top_left_water, top_right_water, bottom_right_water])
# mixing
bottom_mixing = bottom_left + 0.5*door_pos*slope_vec + mixing_pad*slope_vec_up
top_mixing = bottom_mixing + mixing_height*slope_vec_up
bottom_left_mixing = bottom_mixing - 0.5*mixing_width*slope_vec
bottom_right_mixing = bottom_mixing + 0.5*mixing_width*slope_vec

# sediment position generation
ngrains = 70
np.random.seed(220212021)
xsed, ysed = door_pos*np.random.random((ngrains, )), water_height*(1+sind(slope))*np.random.random((ngrains, ))
xsed, ysed = (np.dot(Rotation_matrix(slope), np.array([xsed, ysed])).T - tank_length*slope_vec).T


# #### Figure
figwidth = 0.6*Beamer.fig_width
fig, ax = plt.subplots(1, 1, figsize=(figwidth, 0.5*figwidth),
                       constrained_layout=True)

ax.set_xlim(bottom_left[0]-0.005*tank_length, top_right[0] + 0.005*tank_length)
ax.set_ylim(bottom_right[1] - 0.2*door_height, top_door[1]+0.4*door_height)
plt.axis('off')
ax.set_aspect('equal')
#
# ## tank walls
ax.plot([bottom_left[0], bottom_right[0]], [bottom_left[1], bottom_right[1]], color=color_walls)
ax.plot([bottom_left[0], top_left[0]], [bottom_left[1], top_left[1]], color=color_walls)
ax.plot([bottom_right[0], top_right[0]], [bottom_right[1], top_right[1]], color=color_walls)
ax.plot([bottom_door[0], top_door[0]], [bottom_door[1], top_door[1]], color=color_walls)

# ## water
poly_water = plt.Polygon(xy_water, facecolor=color_water, alpha=alpha_water, edgecolor=None)
ax.add_patch(poly_water)

# ## sediments
ax.scatter(xsed[ysed < water_height], ysed[ysed < water_height], color=color_sed, s=1)

# ## mixing
ax.plot([bottom_mixing[0], top_mixing[0]], [bottom_mixing[1], top_mixing[1]], color=color_mixing)
ax.plot([bottom_left_mixing[0], bottom_right_mixing[0]],
        [bottom_left_mixing[1], bottom_right_mixing[1]], color=color_mixing, lw=2)

# ## annotations
ax.plot([bottom_right[0], bottom_right[0]-0.4*tank_length], [bottom_right[1], bottom_right[1]], ls='--', color='k')
theta = np.linspace(180, 180+slope, 100)
x, y = 0.35*tank_length*np.array([cosd(theta), sind(theta)]) + bottom_right[:, None]
ax.plot(x, y, color='k')
#
ax.text(bottom_right[0]-0.425*tank_length, bottom_right[1] + 0.04*door_height,
        r'$\theta$', ha='right', va='center', fontsize=Beamer.fontsize_small)


ax.annotate("", xytext=top_door, xy=top_door+0.4*door_height*slope_vec_up,
            arrowprops=dict(arrowstyle="-|>", shrinkA=5, shrinkB=5, color='k'))

xy = bottom_right - 0.195*door_height*slope_vec_up
xytext = bottom_door - 0.195*door_height*slope_vec_up
ax.annotate("", xytext=xytext, xy=xy, arrowprops=dict(arrowstyle="<->",
                                                      shrinkA=0, shrinkB=0,
                                                      color='k'))
xytext = (xy + xytext)/2
ax.text(xytext[0], xytext[1] - 0.05*door_height, r'$1.5~\textrm{m}$', ha='center', va='top',
        fontsize=Beamer.fontsize_small)
#
xy = bottom_left - 0.195*door_height*slope_vec_up
xytext = bottom_door - 0.195*door_height*slope_vec_up
ax.annotate("", xytext=xytext, xy=xy, arrowprops=dict(arrowstyle="<->",
                                                      shrinkA=0, shrinkB=0,
                                                      color='k'))
xytext = (xy + xytext)/2
ax.text(xytext[0] - 0.1*door_height, xytext[1] - 0.06*door_height, r'$L_{0} =$\\$10~\textrm{cm}$', ha='center', va='top', ma='right',
        fontsize=Beamer.fontsize_small)
#
xy = top_right_water - (tank_length - door_pos)*np.array([1, 0])/cosd(slope) + 0.03*tank_length*slope_vec
xytext = bottom_door + 0.03*tank_length*slope_vec
ax.annotate("", xytext=xytext, xy=xy, arrowprops=dict(arrowstyle="<->",
                                                      shrinkA=0, shrinkB=0,
                                                      color='k'))
xytext = (xy + xytext)/2
ax.text(xytext[0] + 0.05*door_height, xytext[1], r'$h_{0} = 20~\textrm{cm}$', ha='right',
        va='center', fontsize=Beamer.fontsize_small)

ax.invert_xaxis()
plt.savefig(os.path.join(fig_dir, 'Sketch_lock_release.svg'), dpi=400)
