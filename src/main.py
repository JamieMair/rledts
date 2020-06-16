import ray
from ray import tune
import ray.rllib.agents.sac as sac
from ray.tune.logger import pretty_print
from envs.excursion_env import ExcursionEnv

ray.init()
config = sac.DEFAULT_CONFIG.copy()
config["env"] = ExcursionEnv
config["env_config"] = {"excursion_time": 100, "win_reward": 10.0, "lose_reward": -2.0, "negative_reward": -5.0}
config["eager"] = False
config["Q_model"]["fcnet_hiddens"] = [16, 16]
config["policy_model"]["fcnet_hiddens"] = [16, 16]
config["initial_alpha"] = 1.0
config["optimization"]["entropy_learning_rate"] = 0.0
config["num_gpus"] = 0
config["num_workers"] = 1


tune.run(
    "SAC",
    stop={"episode_reward_mean": 60},
    config=config,
)
ray.shutdown()