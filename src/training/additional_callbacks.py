import ray
from ray import tune
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.sac.sac_tf_policy import SACTFPolicy
from ray.rllib.agents.sac.sac_tf_model import SACTFModel
import tensorflow as tf
import numpy as np
from typing import Dict

class EntropyCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                            policies: Dict[str, Policy],
                            episode: MultiAgentEpisode, **kwargs):
        
        episode.custom_metrics["episode_entropy"] = 0


    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
            episode: MultiAgentEpisode, **kwargs):
        
        if "default_policy" in episode._policies:
            policy: SACTFPolicy = episode._policies["default_policy"]
            if (len(episode._agent_to_last_action) > 0):
                last_action = episode.last_action_for()
                observation = episode.last_observation_for().reshape(1, 2)
                # Reverse the observation to the last state
                observation[0, 0] -= 1.0 if last_action == 0 else -1.0
                observation[0, 1] -= 1.0
                log_p_action = policy.compute_log_likelihoods(actions=np.array([last_action]), obs_batch=observation)

                episode.custom_metrics["episode_entropy"] -= log_p_action

