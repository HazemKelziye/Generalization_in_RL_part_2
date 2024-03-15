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

ppo_config = PPOConfig()
ppo_config = ppo_config.framework("torch")
ppo_config = ppo_config.training(model={"fcnet_hiddens":[64]}, gamma=0.99, lr=0.0003, train_batch_size=5000,
                                 kl_coeff=0.3, clip_param=0.2)
ppo_config = ppo_config.environment(env="MountainCar-v0")
ppo_config = ppo_config.resources(num_gpus=1, num_cpus_for_local_worker=1,
                                  num_gpus_per_worker=0.2, num_cpus_per_worker=1)
ppo_config = ppo_config.rollouts(num_rollout_workers=5)


agent = ppo_config.build()
agent.restore("checkpoints/mountaincar")


###### Evaluation through the environment
env = gym.make("MountainCar-v0", render_mode=None)
terminated = truncated = False
EPISODES_NUM = 6500
sa_pair = []
episodes_dict = {}

for ep_counter in range(1, EPISODES_NUM + 1):
    print(f"Episode {ep_counter}:")
    obs, _ = env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.compute_single_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        sa_pair.append([obs.tolist(), action.tolist()])

    print(len(sa_pair))
    episodes_dict[f"episode{ep_counter}"] = sa_pair
    sa_pair = []


json_object = json.dumps(episodes_dict)
with open("../ENV_datasets/mountaincar_ds.json", "w") as outfile:
    outfile.write(json_object)
ray.shutdown()
