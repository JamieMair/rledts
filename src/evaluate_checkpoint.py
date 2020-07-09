import argparse
import os
from ray.rllib.agents.sac.sac import SACTrainer, SACTFPolicy
from ray.rllib.agents.callbacks import DefaultCallbacks
from training.additional_callbacks import EntropyCallbacks
from envs.excursion_env import ExcursionEnv
import json
import tensorflow as tf
import numpy as np
import ray
import time

parser = argparse.ArgumentParser("Excursion Evaluator")

parser.add_argument("--directory", type=str, help="Directory of the tune experiment.")
parser.add_argument("--checkpoint", type=int, help="The number of the checkpoint saved.")

args = parser.parse_args()


if not os.path.isdir(args.directory):
    print(f"Could not find directory at {args.directory}.")
    exit()
checkpoint_directory = os.path.join(args.directory, f"checkpoint_{args.checkpoint}")

if not os.path.isdir(checkpoint_directory):
    print(f"Could not find directory at {checkpoint_directory}.")
    exit()


params_path = os.path.join(args.directory, "params.json")
print(f"Loading parameters at {params_path}")
with open(params_path, 'r') as reader:
    config = json.load(reader)

print("Config loaded.")
print(f"Restoring checkpoint at {checkpoint_directory}")

ray.init()
config["callbacks"] = EntropyCallbacks
config["env"] = ExcursionEnv
config["num_workers"] = 1
trainer : SACTrainer = SACTrainer(config=config)

checkpoint_path = os.path.join(checkpoint_directory, f"checkpoint-{args.checkpoint}")
trainer.restore(checkpoint_path)

model_directory = os.path.join(checkpoint_directory, "model")

# trainer.export_policy_model(model_directory)

# model_path = os.path.join(model_directory, "saved_model.pb")

# loaded_model = tf.saved_model.load(model_directory)

# observation_tensor = loaded_model.graph.get_operation_by_name("default_policy/observations")
# log_p_tensor = loaded_model.graph.get_operation_by_name("default_policy/action_logp")

# with tf.compat.v1.Session() as sess:
#     with loaded_model.graph.as_default():
#         value = sess.run([log_p_tensor], feed_dict = {observation_tensor: np.array([0, 0]).reshape(2,1)})

# print(loaded_model)

# policy: SACTFPolicy = trainer._policy

min_samples = 50
precision = 0.002
precision_squared = precision*precision
policy_dict = {}
start_time = time.time()
T = config["env_config"]["excursion_time"]
for t in range(T):
    for x in range(-t,t+1, 2):
        observation = [x, t]
        count_up = 0
        count_down = 0

        for i in range(min_samples):
            action = trainer.compute_action(observation)
            if action == 0:
                count_up += 1
            else:
                count_down += 1
        while True:

            for i in range(10):
                action = trainer.compute_action(observation)
                if action == 0:
                    count_up += 1
                else:
                    count_down += 1

            prob_sample = count_up / (count_up + count_down)

            uncertainty_error_squared =  prob_sample*(1-prob_sample)/(count_down+count_up)

            if uncertainty_error_squared < precision_squared:
                break

        policy_dict[(x,t)] = prob_sample
    current_time = time.time()
    elapsed_time = current_time-start_time
    elapsed_proportion = ((t+1)*(t+2)/2)/(T*(T+1)/2)
    estimated_time_to_complete = elapsed_time/elapsed_proportion
    print(f"Finished time {t}\tETC in: {(estimated_time_to_complete-elapsed_time):.0f}s.")

with open("results.csv", 'w') as writer:
    writer.write("x,t,policy_up,policy_down\n")
    for state, probability in policy_dict.items():
        x, t = state
        writer.write(f"{x},{t},{probability}\n")



