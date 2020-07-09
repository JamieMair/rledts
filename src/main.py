import ray
from ray import tune
import ray.rllib.agents.sac as sac
from ray.tune.logger import pretty_print
from envs.excursion_env import ExcursionEnv
from training.additional_callbacks import EntropyCallbacks

ray.init()
config = sac.DEFAULT_CONFIG.copy()
config["env"] = ExcursionEnv
config["env_config"] = {"excursion_time": 100, "win_reward": 0.0, "lose_reward": -10.0, "negative_reward": -5.0}
config["eager"] = False
config["Q_model"]["fcnet_hiddens"] = [32, 32]
config["policy_model"]["fcnet_hiddens"] = [32, 32]
config["initial_alpha"] = 1.0
config["optimization"]["entropy_learning_rate"] = 0.0
config["num_gpus"] = 0
config["gamma"] = 1.0
config["num_workers"] = 16-1
config["callbacks"] = EntropyCallbacks

def stopper(trial_id, result):
    if (result["training_iteration"] < 10000):
        return False
    
    return result["episode_reward_mean"] > -0.01 and result["custom_metrics"]["success_mean"] > 0.95

tune.run(
    "SAC",
    stop=stopper,
    config=config,
    checkpoint_freq=25,
    checkpoint_at_end=True
)
ray.shutdown()