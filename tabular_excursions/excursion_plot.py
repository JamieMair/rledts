import sys
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
source_path = os.path.join("source/")
sys.path.insert(0,source_path)
import excursions # pylint: disable = import-error

trajectory_length = 100
positivity_bias = 5
target_bias = 7
return_learning_rate = 0.1
success_learning_rate = 0.003
entropy_learning_rate = 0.01
policy_learning_rate_1 = 0.05
policy_learning_rate_2 = 0.05
policy_learning_rate_3 = 0.15
value_learning_rate_2 = 0.3
value_learning_rate_3 = 0.3

### Plot data ###

environment_parameters = dict(
	trajectory_length = trajectory_length, 
	positivity_bias = positivity_bias,
	target_bias = target_bias,
)
gauge = excursions.gauge(environment_parameters)

initial_name = ("data/tl%s_pb%s_tb%s_data.npy"%(
	trajectory_length,
	positivity_bias,
	target_bias))
shared_name = ("data/tl%s_pb%s_tb%s_rl%s_sl%s_el%s"%(
	trajectory_length,
	positivity_bias,
	target_bias,
	return_learning_rate,
	success_learning_rate,
	entropy_learning_rate))
agent1_name = ("_%spl_%sAlg_data.npy"%(
	policy_learning_rate_1,
	'mc'))
agent2_name = ("_%spl_%svl_%sAlg_data.npy"%(
	policy_learning_rate_2,
	value_learning_rate_2,
	'mcvb'))
agent3_name = ("_%spl_%svl_%sAlg_data.npy"%(
	policy_learning_rate_3,
	value_learning_rate_3,
	'ac'))

initial_data = np.load(initial_name, allow_pickle=True)
agent1_data = np.load(shared_name + agent1_name, allow_pickle=True)
agent2_data = np.load(shared_name + agent2_name, allow_pickle=True)
agent3_data = np.load(shared_name + agent3_name, allow_pickle=True)

### Plot parameters ###

sample_plot_start = -0.02 * trajectory_length
sample_plot_end = 1.02 * trajectory_length
negative_shade_bound = [sample_plot_start, sample_plot_end]
sample_y_range = np.max(initial_data[0]) - np.min(initial_data[0])
initial_min_y = np.min(initial_data[0]) - sample_y_range*0.02
initial_max_y = np.max(initial_data[0]) + sample_y_range*0.02
sample_min_y = -20 * 1.2
sample_max_y = 20 * 1.2
sample_alpha = 0.2
initial_sample_alpha = 0.1
target_size = 40

episode_min = - 0.035 * len(agent1_data[1]) 
episode_max = 1.035 * len(agent1_data[1]) 

color1 = cm.viridis(0.4)
color2 = cm.viridis(0.7)
color3 = cm.viridis(0.2)

plt.rc('font', size = 20)
#font = {'family' : 'times new roman','serif' : 'times new roman','weight' : 'normal',#'size'   : 15}
#plt.rc('font', **font)
plt.rc('text', usetex = True)
fig = plt.figure(figsize = (12, 6))

### Axes parameters ###

left_spacing = 0.085
plot_width1 = 0.3
plot_wspacing1 = 0.1
plot_width2 = plot_width1
plot_wspacing2 = plot_wspacing1
plot_width3 = plot_width1
colorbar_wspacing = plot_wspacing1 * 0.2
colorbar_width = plot_width3 * 0.04
right_spacing = 0.075
total_width = (left_spacing + plot_width1 + plot_wspacing1 + plot_width2
			   + plot_wspacing2 + plot_width3 + colorbar_width + colorbar_wspacing
			   + right_spacing)

bottom_spacing = 0.07
plot_height1 = 0.3
plot_hspacing1 = 0
plot_height2 = plot_height1
plot_hspacing2 = plot_hspacing1
plot_height3 = plot_height1
top_spacing = 0.015
total_height = (bottom_spacing + plot_height1 + plot_hspacing1 + plot_height2
				+ plot_hspacing2 + plot_height3 + top_spacing)

colorbar_height = 0.85 * plot_height1

label_x_shift_scale = 0.26
label_y_shift_scale = -0.13

### Plot 1 ###

returns_x_shift_scale = 0.6
initial_y_shift_scale = 0.65
agent_y_shift_scale = 0.15

x_position = left_spacing
y_position = (bottom_spacing + plot_height1 + plot_hspacing1 
			  + plot_height2 + plot_hspacing2)
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height3*(1 + label_y_shift_scale))/total_height,
	"(a)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width1/total_width,
	plot_height3/total_height))
plt.plot(agent3_data[1], c = color3)
plt.plot(agent1_data[1], c = color1)
plt.plot(agent2_data[1], c = color2)
plt.plot([episode_min, episode_max], [gauge.max_return, gauge.max_return], 
		 ls = '--', c = 'k', alpha = 0.5, lw = 1)
plt.xlim(episode_min, episode_max)
plt.setp(ax.get_xticklabels(), visible = False)
plt.yticks([-350,0],[r'$-350$',r'$0$'])
plt.ylabel(r'$\left\langle R\right\rangle$', labelpad = -17)
plt.ylim(-400,10)
plt.figtext(
	(x_position + returns_x_shift_scale*plot_width1)/total_width,
	(y_position + (initial_y_shift_scale)*plot_height1)/total_height,
	round(initial_data[1], 2))
plt.figtext(
	(x_position + returns_x_shift_scale*plot_width1)/total_width,
	((y_position + (initial_y_shift_scale - agent_y_shift_scale)*plot_height1)
	 / total_height),
	round(agent1_data[7], 2), color = color1)
plt.figtext(
	(x_position + returns_x_shift_scale*plot_width1)/total_width,
	((y_position + (initial_y_shift_scale - agent_y_shift_scale*2)*plot_height1)
	 / total_height),
	round(agent2_data[7], 2), color = color2)
plt.figtext(
	(x_position + returns_x_shift_scale*plot_width1)/total_width,
	((y_position + (initial_y_shift_scale - agent_y_shift_scale*3)*plot_height1)
	 / total_height),
	round(agent3_data[7], 2), color = color3)
plt.figtext(
	(x_position + returns_x_shift_scale*plot_width1)/total_width,
	((y_position + (initial_y_shift_scale - agent_y_shift_scale*4)*plot_height1)
	 / total_height),
	round(gauge.max_return, 2))

### Plot 2 ###

x_position = left_spacing
y_position = bottom_spacing + plot_height1 + plot_hspacing1
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height3*(1 + label_y_shift_scale))/total_height,
	"(b)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width1/total_width,
	plot_height2/total_height))
plt.plot(agent3_data[2], c = color3)
plt.plot(agent1_data[2], c = color1)
plt.plot(agent2_data[2], c = color2)
plt.plot([episode_min, episode_max], [1, 1], 
		 ls = '--', c = 'k', alpha = 0.5, lw = 1)
plt.xlim(episode_min, episode_max)
plt.setp(ax.get_xticklabels(), visible = False)
plt.yticks([0,0.5,1],[r'$0$',r'$0.5$',r'$1$'])
plt.ylabel(r'$\left\langle S\right\rangle$', labelpad = 3)

### Plot 3 ###

x_position = left_spacing
y_position = bottom_spacing
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height3*(1 + label_y_shift_scale))/total_height,
	"(c)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width1/total_width,
	plot_height1/total_height))
plt.plot(agent3_data[3], c = color3)
plt.plot(agent1_data[3], c = color1)
plt.plot(agent2_data[3], c = color2)
plt.plot([episode_min, episode_max], 
		 [trajectory_length * np.log(2), trajectory_length * np.log(2)], 
		 ls = '--', c = 'k', alpha = 0.5, lw = 1)
plt.xlim(episode_min, episode_max)
plt.xticks([0,50000],['0','50000'])
plt.xlabel("Episode", labelpad = -15)
plt.yticks([20,40,60],[r'$20$',r'$40$',r'$60$'])
plt.ylabel(r'$\left\langle \mathcal{H}\right\rangle$', labelpad = 8)

### Plot 4 ###

x_position = left_spacing + plot_width1 + plot_wspacing1
y_position = (bottom_spacing + plot_height1 + plot_hspacing1 
			  + plot_height2 + plot_hspacing2)
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height3*(1 + label_y_shift_scale))/total_height,
	"(d)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width2/total_width,
	plot_height3/total_height))
plt.plot(initial_data[0], c = 'k', alpha = initial_sample_alpha)
plt.plot(agent1_data[4], c = color1, alpha = sample_alpha)
plt.scatter([trajectory_length], 
			[0], c = 'k', marker = 'o', s = target_size)
plt.plot([0, sample_max_y], [0, sample_max_y], lw = 1, ls = '--', c = 'k', alpha = 0.5)
plt.plot([0, -sample_min_y], [0, sample_min_y], lw = 1, ls = '--', c = 'k', alpha = 0.5)
plt.plot(negative_shade_bound, [0, 0], lw = 2, c = 'r', ls = '--', alpha = 0.3)
plt.fill_between(negative_shade_bound, [0, 0], 
				 [sample_min_y, sample_min_y], color = 'r', alpha = 0.1)
plt.xlim(sample_plot_start, sample_plot_end)
plt.ylim(sample_min_y, sample_max_y)
plt.setp(ax.get_xticklabels(), visible = False)
plt.ylabel(r"$x$", labelpad = -20)

### Plot 5 ###

x_position = left_spacing + plot_width1 + plot_wspacing1
y_position = bottom_spacing + plot_height1 + plot_hspacing1
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height3*(1 + label_y_shift_scale))/total_height,
	"(e)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width2/total_width,
	plot_height2/total_height))
plt.plot(initial_data[0], c = 'k', alpha = initial_sample_alpha)
plt.plot(agent2_data[4], c = color2, alpha = sample_alpha)
plt.scatter([trajectory_length], 
			[0], c = 'k', marker = 'o', s = target_size)
plt.plot([0, sample_max_y], [0, sample_max_y], lw = 1, ls = '--', c = 'k', alpha = 0.5)
plt.plot([0, -sample_min_y], [0, sample_min_y], lw = 1, ls = '--', c = 'k', alpha = 0.5)
plt.plot(negative_shade_bound, [0, 0], lw = 2, c = 'r', ls = '--', alpha = 0.3)
plt.fill_between(negative_shade_bound, [0, 0], 
				 [sample_min_y, sample_min_y], color = 'r', alpha = 0.1)
plt.xlim(sample_plot_start, sample_plot_end)
plt.ylim(sample_min_y, sample_max_y)
plt.setp(ax.get_xticklabels(), visible = False)
plt.ylabel(r"$x$", labelpad = -20)

### Plot 6 ###

x_position = left_spacing + plot_width1 + plot_wspacing1
y_position = bottom_spacing
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height3*(1 + label_y_shift_scale))/total_height,
	"(f)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width2/total_width,
	plot_height1/total_height))
plt.plot(initial_data[0], c = 'k', alpha = initial_sample_alpha)
plt.plot(agent3_data[4], c = color3, alpha = sample_alpha)
plt.scatter([trajectory_length], 
			[0], c = 'k', marker = 'o', s = target_size)
plt.plot([0, sample_max_y], [0, sample_max_y], lw = 1, ls = '--', c = 'k', alpha = 0.5)
plt.plot([0, -sample_min_y], [0, sample_min_y], lw = 1, ls = '--', c = 'k', alpha = 0.5)
plt.plot(negative_shade_bound, [0, 0], lw = 2, c = 'r', ls = '--', alpha = 0.3)
plt.fill_between(negative_shade_bound, [0, 0], 
				 [sample_min_y, sample_min_y], color = 'r', alpha = 0.1)
plt.xlim(sample_plot_start, sample_plot_end)
plt.ylim(sample_min_y, sample_max_y)
plt.xticks([0,100],['0','100'])
plt.xlabel(r"$t$", labelpad = -15)
plt.ylabel(r"$x$", labelpad = -20)

### Plot calculations ###

up_probabilities3 = 1/(1+np.exp(-agent3_data[5]))
up_probabilities = np.concatenate((up_probabilities3[trajectory_length + 1 : 
											 2*trajectory_length + 1],
								  up_probabilities3[0 : trajectory_length + 1]))
state_probabilities = agent3_data[6]
negated_values = -np.concatenate((agent3_data[10][trajectory_length + 1 : 
										   2*trajectory_length + 1],
								 agent3_data[10][0 : trajectory_length + 1]))
positions = np.arange(201) - 100
times = np.arange(101)
for t in range(101):
	for x in range(201):
		if t % 2 == 0:
			if x % 2 == 1:
				up_probabilities[x][t] = 0.25*(up_probabilities[x][(t+1)%100] 
											   + up_probabilities[x][(t-1)%100]
											   + up_probabilities[(x+1)%200][t]
											   + up_probabilities[(x-1)%200][t])
				state_probabilities[x][t] = 0.25*(state_probabilities[x][(t+1)%100] 
												  + state_probabilities[x][(t-1)%100]
											   	  + state_probabilities[(x+1)%200][t]
											   	  + state_probabilities[(x-1)%200][t])
				negated_values[x][t] = 0.25*(negated_values[x][(t+1)%100] 
											 + negated_values[x][(t-1)%100]
											 + negated_values[(x+1)%200][t]
											 + negated_values[(x-1)%200][t])
		if t % 2 == 1:
			if x % 2 == 0:
				up_probabilities[x][t] = 0.25*(up_probabilities[x][(t+1)%100] 
											   + up_probabilities[x][(t-1)%100]
											   + up_probabilities[(x+1)%200][t]
											   + up_probabilities[(x-1)%200][t])
				state_probabilities[x][t] = 0.25*(state_probabilities[x][(t+1)%100] 
										 		  + state_probabilities[x][(t-1)%100]
										 		  + state_probabilities[(x+1)%200][t]
										 		  + state_probabilities[(x-1)%200][t])
				negated_values[x][t] = 0.25*(negated_values[x][(t+1)%100] 
											 + negated_values[x][(t-1)%100]
											 + negated_values[(x+1)%200][t]
											 + negated_values[(x-1)%200][t])
times, positions = np.meshgrid(times, positions)

### Plot 7 ###

x_position = (left_spacing + plot_width1 + plot_wspacing1 + plot_width2
			  + plot_wspacing2)
y_position = (bottom_spacing + plot_height1 + plot_hspacing1 
			  + plot_height2 + plot_hspacing2)
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height3*(1 + label_y_shift_scale))/total_height,
	"(g)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width3/total_width,
	plot_height3/total_height))
plt.pcolormesh(times, positions, up_probabilities,
		   norm = LogNorm(vmin = 10**(-1), vmax = 1), antialiased = True)
plt.plot([0,sample_max_y], [0, sample_max_y], lw = 2, c = 'k', alpha = 1)
plt.fill_between([0,sample_max_y], [0, sample_max_y], [sample_max_y, sample_max_y], 
				 color = 'w', alpha = 1)
plt.plot([0,sample_max_y], [0, sample_min_y], lw = 2, c = 'k', alpha = 1)
plt.fill_between([0,sample_max_y], [0, sample_min_y], [sample_min_y, sample_min_y], 
				 color = 'w', alpha = 1)
plt.ylim(sample_min_y, sample_max_y)
plt.setp(ax.get_xticklabels(), visible = False)
plt.ylabel(r"$x$", labelpad = -20)

x_position += plot_width3 + colorbar_wspacing
y_position += 0.5 * (plot_height1 - colorbar_height)
ax1 = plt.axes((
	x_position/total_width,
	y_position/total_height,
	colorbar_width/total_width,
	colorbar_height/total_height))
cb1 = matplotlib.colorbar.ColorbarBase(ax1,
                                norm=LogNorm(vmin = 10**(-1), vmax = 10**(1)),
                                orientation='vertical', ticks = [10**(-1),10**(1)])
cb1.set_ticklabels(ticklabels = [r'${10}^{-1}$',r'${10}^{1}$'])
cb1.set_label(r'$P_\theta(1|x,t)$', labelpad = -15)

### Plot 8 ###

x_position = (left_spacing + plot_width1 + plot_wspacing1 + plot_width2
			  + plot_wspacing2)
y_position = bottom_spacing + plot_height1 + plot_hspacing1
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height3*(1 + label_y_shift_scale))/total_height,
	"(h)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width3/total_width,
	plot_height2/total_height))
plt.pcolormesh(times, positions, negated_values,
		   norm = LogNorm(vmin = 10**(0), vmax = 10**2), antialiased = True)
plt.plot([0,sample_max_y], [0, sample_max_y], lw = 2, c = 'k', alpha = 1)
plt.fill_between([0,sample_max_y], [0, sample_max_y], [sample_max_y, sample_max_y], 
				 color = 'w', alpha = 1)
plt.plot([0,sample_max_y], [0, sample_min_y], lw = 2, c = 'k', alpha = 1)
plt.fill_between([0,sample_max_y], [0, sample_min_y], [sample_min_y, sample_min_y], 
				 color = 'w', alpha = 1)
plt.ylim(sample_min_y, sample_max_y)
plt.setp(ax.get_xticklabels(), visible = False)
plt.ylabel(r"$x$", labelpad = -20)

x_position += plot_width3 + colorbar_wspacing
y_position += 0.5 * (plot_height1 - colorbar_height)
ax1 = plt.axes((
	x_position/total_width,
	y_position/total_height,
	colorbar_width/total_width,
	colorbar_height/total_height))
cb1 = matplotlib.colorbar.ColorbarBase(ax1,
                                norm=LogNorm(vmin = 10**(0), vmax = 10**(2)),
                                orientation='vertical', ticks = [10**(0),10**(2)])
cb1.set_ticklabels(ticklabels = [r'${10}^{0}$',r'${10}^{2}$'])
cb1.set_label(r'$V_\psi(x,t)$', labelpad = -6.5)

### Plot 9 ###

x_position = (left_spacing + plot_width1 + plot_wspacing1 + plot_width2
			  + plot_wspacing2)
y_position = bottom_spacing
plt.figtext(
	(x_position - plot_width1*label_x_shift_scale)/total_width,
	(y_position + plot_height3*(1 + label_y_shift_scale))/total_height,
	"(i)")
ax = plt.axes((
	x_position/total_width,
	y_position/total_height,
	plot_width3/total_width,
	plot_height1/total_height))
plt.pcolormesh(times, positions, state_probabilities, 
			   norm = LogNorm(vmin = 10**(-3), vmax = 1), antialiased = True)
plt.plot([0,sample_max_y], [0, sample_max_y], lw = 2, c = 'k', alpha = 1)
plt.fill_between([0,sample_max_y], [0, sample_max_y], [sample_max_y, sample_max_y], 
				 color = 'w', alpha = 1)
plt.plot([0,sample_max_y], [0, sample_min_y], lw = 2, c = 'k', alpha = 1)
plt.fill_between([0,sample_max_y], [0, sample_min_y], [sample_min_y, sample_min_y], 
				 color = 'w', alpha = 1)
plt.ylim(sample_min_y, sample_max_y)
plt.xticks([0,100],['0','100'])
plt.xlabel(r"$t$", labelpad = -15)
plt.ylabel(r"$x$", labelpad = -20)

x_position += plot_width3 + colorbar_wspacing
y_position += 0.5 * (plot_height1 - colorbar_height)
ax1 = plt.axes((
	x_position/total_width,
	y_position/total_height,
	colorbar_width/total_width,
	colorbar_height/total_height))
cb1 = matplotlib.colorbar.ColorbarBase(ax1,
                                norm = LogNorm(vmin = 10**(-3), vmax = 1),
                                orientation='vertical', ticks = [10**(-3),10**(0)])
cb1.set_ticklabels(ticklabels = [r'${10}^{-3}$',r'${10}^{0}$'])
cb1.set_label(r'$P_\theta(x|t)$', labelpad = -15)


fig.savefig('tabular_excursions.pdf')
plt.show()