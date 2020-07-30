import sys
import os
import numpy as np
from matplotlib import pyplot as plt
source_path = os.path.join("source/")
sys.path.insert(0,source_path)
import excursions # pylint: disable = import-error
import algorithms # pylint: disable = import-error
import tables # pylint: disable = import-error

environment_parameters = dict(
	trajectory_length = 1000, 
	positivity_bias = 5,
	target_bias = 7,
)

gauge = excursions.gauge(environment_parameters)
print(gauge.max_return)
print(gauge.max_ent_return)
print()

environment = excursions.environment(environment_parameters)
success_learning_rate = 0.003
entropy_learning_rate = 0.01
analyser1 = excursions.analysis(success_learning_rate, entropy_learning_rate)
analyser2 = excursions.analysis(success_learning_rate, entropy_learning_rate)
analyser3 = excursions.analysis(success_learning_rate, entropy_learning_rate)

table_dimension = (environment_parameters['trajectory_length']*2 + 1, 
				   environment_parameters['trajectory_length'] + 1)
policy1 = tables.two_action_policy_table(table_dimension, 0.05)
values2 = tables.value_table(table_dimension, 0.3)
policy2 = tables.two_action_policy_table(table_dimension, 0.05)
values3 = tables.value_table(table_dimension, 0.3)
policy3 = tables.two_action_policy_table(table_dimension, 0.15)

algorithm_parameters1 = dict(
	environment = environment, 
	return_learning_rate = 0.1,
	policy = policy1,
	analyser = analyser1,
)
algorithm_parameters2 = dict(
	environment = environment, 
	return_learning_rate = 0.1,
	values = values2,
	policy = policy2,
	analyser = analyser2,
)
algorithm_parameters3 = dict(
	environment = environment, 
	return_learning_rate = 0.1,
	values = values3,
	policy = policy3,
	analyser = analyser3,
)
agent1 = algorithms.kl_regularized_monte_carlo_returns(algorithm_parameters1)
agent2 = algorithms.kl_regularized_monte_carlo_value_baseline(algorithm_parameters2)
agent3 = algorithms.kl_regularized_actor_critic(algorithm_parameters3)


agent1.evaluate(10000)
initial_return = agent1.average_return
initial_success = agent1.analyser.average_success
initial_entropy = agent1.analyser.average_entropy
agent2.average_return = initial_return
agent3.average_return = initial_return
agent2.analyser.average_success = initial_success
agent3.analyser.average_success = initial_success
agent2.analyser.average_entropy = initial_entropy
agent3.analyser.average_entropy = initial_entropy
print("Initial return: %s"%(initial_return))
initial_samples = agent1.samples(50)

min_y = np.min(np.array(initial_samples)[:,:,0]) - 1
max_y = np.max(np.array(initial_samples)[:,:,0]) + 1

episodes = 200000
agent1.train(episodes)
agent2.train(episodes)
agent3.train(episodes)

evals = 1000
agent1.evaluate(evals)
agent2.evaluate(evals)
agent3.evaluate(evals)
final_return1 = agent1.average_return
final_return2 = agent2.average_return
final_return3 = agent3.average_return
print("Initial return: %s, agent1's final return: %s, agent2's final return: %s, agent3's final return: %s"
%(initial_return, final_return1, final_return2, final_return3))
samples1 = agent1.samples(50)
samples2 = agent2.samples(50)
samples3 = agent3.samples(50)

traj_len = environment_parameters['trajectory_length']
plot_end = traj_len + 1

up_probabilities1 = 1/(1+np.exp(-agent1.policy.table))
state_probabilities1 = excursions.state_probabilities(up_probabilities1, traj_len)
up_probabilities2 = 1/(1+np.exp(-agent2.policy.table))
state_probabilities2 = excursions.state_probabilities(up_probabilities2, traj_len)
up_probabilities3 = 1/(1+np.exp(-agent3.policy.table))
state_probabilities3 = excursions.state_probabilities(up_probabilities3, traj_len)

initial_data = [np.array(initial_samples)[:,:,0].T,
				initial_return,
				initial_success,
				initial_entropy]
agent1_data = [agent1.returns, 
			   agent1.average_returns, 
			   agent1.analyser.successes, 
			   agent1.analyser.entropies,
			   np.array(samples1)[:,:,0].T,
			   agent1.policy.table,
			   state_probabilities1,
			   final_return1,
			   agent1.analyser.average_success,
			   agent1.analyser.average_entropy]
agent2_data = [agent2.returns, 
			   agent2.average_returns, 
			   agent2.analyser.successes, 
			   agent2.analyser.entropies,
			   np.array(samples2)[:,:,0].T,
			   agent2.policy.table,
			   state_probabilities2,
			   final_return2,
			   agent2.analyser.average_success,
			   agent2.analyser.average_entropy,
			   agent2.values.table]
agent3_data = [agent3.returns, 
			   agent3.average_returns, 
			   agent3.analyser.successes, 
			   agent3.analyser.entropies,
			   np.array(samples3)[:,:,0].T,
			   agent3.policy.table,
			   state_probabilities3,
			   final_return3,
			   agent3.analyser.average_success,
			   agent3.analyser.average_entropy,
			   agent3.values.table]

save_initial = False
if save_initial:
	initial_name = ("data/tl%s_pb%s_tb%s_data"%(
		environment_parameters['trajectory_length'],
		environment_parameters['positivity_bias'],
		environment_parameters['target_bias']))
	np.save(initial_name, initial_data)

save = False
if save:
	shared_name = ("data/tl%s_pb%s_tb%s_rl%s_sl%s_el%s"%(
		environment_parameters['trajectory_length'],
		environment_parameters['positivity_bias'],
		environment_parameters['target_bias'],
		algorithm_parameters1['return_learning_rate'],
		success_learning_rate,
		entropy_learning_rate))
	agent1_name = ("_%spl_%sAlg_"%(
		agent1.policy.learning_rate,
		'mc'))
	agent2_name = ("_%spl_%svl_%sAlg_"%(
		agent2.policy.learning_rate,
		agent2.values.learning_rate,
		'mcvb'))
	agent3_name = ("_%spl_%svl_%sAlg_"%(
		agent3.policy.learning_rate,
		agent3.values.learning_rate,
		'ac'))
	np.save(shared_name + agent1_name + "data", agent1_data)
	np.save(shared_name + agent2_name + "data", agent2_data)
	np.save(shared_name + agent3_name + "data", agent3_data)

plt.figure(figsize = (12, 9.5))

plt.subplot(331)
plt.plot(agent3.average_returns, c = 'g')
plt.plot(agent1.average_returns, c = 'b')
plt.plot(agent2.average_returns, c = 'm')
plt.xlabel("Episode")
plt.ylabel("Running return")

plt.subplot(332)
plt.plot(agent3.analyser.successes, c = 'g')
plt.plot(agent1.analyser.successes, c = 'b')
plt.plot(agent2.analyser.successes, c = 'm')
plt.xlabel("Episode")
plt.ylabel("Successes")

plt.subplot(333)
plt.plot(agent3.analyser.entropies, c = 'g')
plt.plot(agent1.analyser.entropies, c = 'b')
plt.plot(agent2.analyser.entropies, c = 'm')
plt.xlabel("Episode")
plt.ylabel("Entropy")

plt.subplot(334)
plt.plot(np.array(initial_samples)[:,:,0].T, c = 'k', alpha = 0.2)
plt.plot(np.array(samples1)[:,:,0].T, c = 'b', alpha = 0.2)
plt.scatter([environment_parameters['trajectory_length']], 
			[0], c = 'k', marker = 'o', s = 80)
plt.plot([-1, plot_end], [0, 0], lw = 2, c = 'r', ls = '--', alpha = 0.3)
plt.fill_between([-1, plot_end], [0, 0], [min_y, min_y], color = 'r', alpha = 0.1)
plt.xlim(-1, plot_end)
plt.ylim(min_y, max_y)
plt.title("Agent 1")
plt.xlabel("Time")
plt.ylabel("Position")

plt.subplot(335)
plt.plot(np.array(initial_samples)[:,:,0].T, c = 'k', alpha = 0.2)
plt.plot(np.array(samples2)[:,:,0].T, c = 'm', alpha = 0.2)
plt.scatter([environment_parameters['trajectory_length']], 
			[0], c = 'k', marker = 'o', s = 80)
plt.plot([-1, plot_end], [0, 0], lw = 2, c = 'r', ls = '--', alpha = 0.3)
plt.fill_between([-1, plot_end], [0, 0], [min_y, min_y], color = 'r', alpha = 0.1)
plt.xlim(-1, plot_end)
plt.ylim(min_y, max_y)
plt.title("Agent 2")
plt.xlabel("Time")

plt.subplot(336)
plt.plot(np.array(initial_samples)[:,:,0].T, c = 'k', alpha = 0.2)
plt.plot(np.array(samples3)[:,:,0].T, c = 'g', alpha = 0.2)
plt.scatter([environment_parameters['trajectory_length']], 
			[0], c = 'k', marker = 'o', s = 80)
plt.plot([-1, plot_end], [0, 0], lw = 2, c = 'r', ls = '--', alpha = 0.3)
plt.fill_between([-1, plot_end], [0, 0], [min_y, min_y], color = 'r', alpha = 0.1)
plt.xlim(-1, plot_end)
plt.ylim(min_y, max_y)
plt.title("Agent 3")
plt.xlabel("Time")

"""
plt.subplot(337)
plt.pcolor(np.concatenate((up_probabilities3[traj_len + 1 : 2*traj_len + 1],
						   up_probabilities3[0 : traj_len + 1])))
plt.colorbar()

plt.subplot(338)
plt.pcolor(state_probabilities3)
plt.colorbar()

plt.subplot(339)
plt.pcolor(np.concatenate((agent3.values.table[traj_len + 1 : 2*traj_len + 1],
						   agent3.values.table[0 : traj_len + 1])))
plt.colorbar()
"""

plt.show()