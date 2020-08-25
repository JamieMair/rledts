import sys
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

scgf = np.load("data/run_3/scgf.npy")
entropy = np.load("data/run_3/entropy.npy")
current = np.load("data/run_3/current.npy")
scgf_discount = np.load("data/run_1/scgf.npy")
entropy_discount = np.load("data/run_1/entropy.npy")
current_discount = np.load("data/run_1/current.npy")
stat_states = np.load("data/run_3/stationary_state.npy")
potentials = np.load("data/run_3/potential.npy")
values = np.load("data/run_3/values.npy")

scgf_exact = np.loadtxt("data/L499.txt").T[1]

initial_bias = -0.25
bias_step = 0.01
biases = [initial_bias + bias_step*i for i in range(101)]
bias_step = 0.05
sparse_biases = [initial_bias + bias_step*i for i in range(21)]
position = [i for i in range(1000)]
bias_mesh, position_mesh = np.meshgrid(biases, position)

### Plot parameters ###

color1 = cm.viridis(0.4)
color2 = cm.viridis(0.7)
color3 = cm.viridis(0.2)

plt.rc('font', size = 20)
plt.rc('text', usetex = True)
fig = plt.figure(figsize = (12, 4.5))

### Axes parameters ###

left_spacing = 0.11
plot_width1 = 0.3
plot_wspacing1 = 0.23
plot_width2 = plot_width1
plot_wspacing2 = plot_wspacing1
plot_width3 = plot_width1
colorbar_wspacing = plot_wspacing1 * 0.1
colorbar_width = plot_width3 * 0.04
right_spacing = 0.075
total_width = (left_spacing + plot_width1 + plot_wspacing1 + plot_width2
			   + plot_wspacing2 + plot_width3 + colorbar_width + colorbar_wspacing
			   + right_spacing)

bottom_spacing = 0.07
plot_height1 = 0.3
plot_hspacing1 = 0.1
plot_height2 = plot_height1
top_spacing = 0.03
total_height = (bottom_spacing + plot_height1 + plot_hspacing1 + plot_height2
				+ top_spacing)

colorbar_height = 0.85 * plot_height1

label_x_shift_scale = 0.35
label_y_shift_scale = -0.05


### Plot 1 ###
x_position = left_spacing
y_position = bottom_spacing + plot_height1 + plot_hspacing1
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height1*(1 + label_y_shift_scale))/total_height,
	"(a)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width1/total_width,
	plot_height2/total_height))
plt.plot([-0.25, 0.75], [0, 0], c = '0.2', lw = 1, ls = ':')
plt.plot(biases, scgf_discount, c = color2, lw = 2.5)
plt.plot(biases, scgf, c = color3, lw = 2.5)
plt.plot(biases, scgf_exact, c = '0.8', lw = 1.5, ls = '--')
#plt.xticks([-0.25, 0.75])
plt.yticks([0.0, 0.2])
plt.ylabel(r'$\bar{r}_\theta(s)$', labelpad = -20)
plt.xlabel(r'$s$', labelpad = -20)

### Plot 2 ###
x_position = left_spacing + plot_width1 + plot_wspacing1
y_position = bottom_spacing + plot_height1 + plot_hspacing1
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height1*(1 + label_y_shift_scale))/total_height,
	"(b)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width2/total_width,
	plot_height2/total_height))
plt.plot([-0.25, 0.75], [np.log(2), np.log(2)], c = '0.2', lw = 1, ls = ':')
plt.plot(biases, entropy_discount, c = color2, lw = 2.5)
plt.plot(biases, entropy, c = color3, lw = 2.5)
plt.xticks([-0.25, 0.75])
plt.yticks([0.5, 0.7])
plt.ylabel(r'$h(s)$', labelpad = -20)
plt.xlabel(r'$s$', labelpad = -20)

### Plot 3 ###
x_position = left_spacing + plot_width1 + plot_wspacing1 + plot_width2 + plot_wspacing2
y_position = bottom_spacing + plot_height1 + plot_hspacing1
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height1*(1 + label_y_shift_scale))/total_height,
	"(c)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width3/total_width,
	plot_height2/total_height))
plt.plot([-0.25, 0.75], 
		 [2*(current[int(len(current)/4+1)]-0.5), 2*(current[int(len(current)/4+1)]-0.5)],
		 c = '0.2', lw = 1, ls = ':')
plt.plot(biases, 2*(current_discount - 0.5), c = color2, lw = 2.5)
plt.plot(biases, 2*(current - 0.5), c = color3, lw = 2.5)
plt.xticks([-0.25, 0.75])
plt.yticks([-0.6, 0.4])
plt.ylabel(r'$v(s)$', labelpad = -20)
plt.xlabel(r'$s$', labelpad = -20)

### Plot 4 ###
x_position = left_spacing
y_position = bottom_spacing
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height1*(1 + label_y_shift_scale))/total_height,
	"(d)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width1/total_width,
	plot_height1/total_height))
plt.pcolormesh(bias_mesh, position_mesh, np.abs(stat_states).T,
			   norm = LogNorm(vmin = 10**(-4), vmax = 10**(-1)), antialiased = True)
plt.xticks([-0.25, 0.75])
plt.yticks([0, 999])
plt.xlabel(r'$s$', labelpad = -20)
plt.ylabel(r'$x$', labelpad = -25)

x_position += plot_width3 + colorbar_wspacing
y_position += 0.5 * (plot_height1 - colorbar_height)
ax1 = plt.axes((
	x_position/total_width,
	y_position/total_height,
	colorbar_width/total_width,
	colorbar_height/total_height))
cb1 = matplotlib.colorbar.ColorbarBase(ax1,
                                norm = LogNorm(vmin = 10**(-4), vmax = 10**(-1)),
                                orientation='vertical', ticks = [10**(-4),10**(-1)])
cb1.set_ticklabels(ticklabels = [r'${10}^{-4}$',r'${10}^{-1}$'])
cb1.set_label(r'$P_\theta(x)$', labelpad = -30)

### Plot 5 ###
x_position = left_spacing + plot_width1 + plot_wspacing1
y_position = bottom_spacing
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height1*(1 + label_y_shift_scale))/total_height,
	"(e)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width2/total_width,
	plot_height1/total_height))
plt.pcolormesh(bias_mesh, position_mesh, potentials.T, vmin = -1.4, vmax = 0.8)
plt.xticks([-0.25, 0.75])
plt.yticks([0, 999])
plt.xlabel(r'$s$', labelpad = -20)
plt.ylabel(r'$x$', labelpad = -25)

x_position += plot_width3 + colorbar_wspacing
y_position += 0.5 * (plot_height1 - colorbar_height)
ax1 = plt.axes((
	x_position/total_width,
	y_position/total_height,
	colorbar_width/total_width,
	colorbar_height/total_height))
cb1 = matplotlib.colorbar.ColorbarBase(ax1, 
								norm = colors.Normalize(vmin = -1.4, vmax = 0.8),
                                orientation='vertical', ticks = [-1.4, 0.8])
cb1.set_ticklabels(ticklabels = [r'$-1.4$',r'$0.8$'])
cb1.set_label(r'$U(x)$', labelpad = -35)

### Plot 6 ###
x_position = left_spacing + plot_width1 + plot_wspacing1 + plot_width2 + plot_wspacing2
y_position = bottom_spacing
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height1*(1 + label_y_shift_scale))/total_height,
	"(f)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width3/total_width,
	plot_height1/total_height))
plt.pcolormesh(bias_mesh, position_mesh, values.T, vmin = -40, vmax = 42)
plt.xticks([-0.25, 0.75])
plt.yticks([0, 999])
plt.xlabel(r'$s$', labelpad = -20)
plt.ylabel(r'$x$', labelpad = -25)

x_position += plot_width3 + colorbar_wspacing
y_position += 0.5 * (plot_height1 - colorbar_height)
ax1 = plt.axes((
	x_position/total_width,
	y_position/total_height,
	colorbar_width/total_width,
	colorbar_height/total_height))
cb1 = matplotlib.colorbar.ColorbarBase(ax1, 
								norm = colors.Normalize(vmin = -40, vmax = 42),
                                orientation='vertical', ticks = [-40, 42])
cb1.set_ticklabels(ticklabels = [r'$-40$',r'$42$'])
cb1.set_label(r'$V_\psi(x)$', labelpad = -30)

fig.savefig('fourier_ring.png', dpi = 800)
plt.show()