import gymnasium as gym
import yaml
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
import GPUtil
import numpy as np
import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPO, PPOConfig
import warnings
import json
from functions import discretize_actions

warnings.filterwarnings("ignore", category=DeprecationWarning)

ray.init(num_cpus=6, num_gpus=1)

sac_config = SACConfig()
sac_config = sac_config.framework("torch")
sac_config = sac_config.training(gamma=0.95, lr=0.01, train_batch_size=32)
sac_config = sac_config.environment(env="Pendulum-v1")
sac_config = sac_config.resources(num_gpus=0)
sac_config = sac_config.rollouts(num_rollout_workers=1)

agent = sac_config.build()
agent.restore("checkpoints/pendulum_sac_2")


###### Evaluation through the environment
env = gym.make("Pendulum-v1", render_mode="human")
terminated = truncated = False
EPISODES_NUM = 2
sa_pair = []
episodes_dict = {}

for ep_counter in range(1, EPISODES_NUM + 1):
    print(f"Episode {ep_counter}:")
    obs, _ = env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.compute_single_action(obs)
        action = np.round(discretize_actions(action), 2)
        obs, reward, terminated, truncated, _ = env.step(action)
        sa_pair.append([obs.tolist(), action.tolist()])

    print(len(sa_pair))
    episodes_dict[f"episode{ep_counter}"] = sa_pair
    sa_pair = []
#
#
# print(type(episodes_dict))
# json_object = json.dumps(episodes_dict)
# with open("env_datasets/test.json", "w") as outfile:
#     outfile.write(json_object)
# ray.shutdown()
