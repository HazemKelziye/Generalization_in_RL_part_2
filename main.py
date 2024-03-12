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

# ppo_config = PPOConfig()
# ppo_config = ppo_config.framework("torch")
# ppo_config = ppo_config.training(model={"fcnet_hiddens": [256, 256]},
#                                  lr=0.0003, kl_coeff=0.3, clip_param=0.2, lambda_=0.95,
#                                  gamma=0.99, train_batch_size=5000)
# ppo_config = ppo_config.environment(env="Acrobot-v1")
# ppo_config = ppo_config.resources(num_gpus=1, num_cpus_for_local_worker=2,
#                                   num_gpus_per_worker=0.5, num_cpus_per_worker=2)
# ppo_config = ppo_config.rollouts(num_rollout_workers=2)

ppo_config = SACConfig()
ppo_config = ppo_config.framework("torch")
ppo_config = ppo_config.training(gamma=0.95, lr=0.01, train_batch_size=32)
ppo_config = ppo_config.environment(env="Pendulum-v1")
ppo_config = ppo_config.resources(num_gpus=0)
ppo_config = ppo_config.rollouts(num_rollout_workers=1)


agent = ppo_config.build()
agent.restore("checkpoints/pendulum_sac_2")

# agent = ppo_config.build()
# agent.restore("/home/basel/PycharmProjects/pythonProject/checkpoints/checkpoints6")


###### Evaluation through the environment
env = gym.make("Pendulum-v1", render_mode="human")
terminated = truncated = False
EPISODES_NUM = 3
sa_pair = []
episodes_dict = {}

for ep_counter in range(1, EPISODES_NUM + 1):
    print(f"Episode {ep_counter}:")
    obs, _ = env.reset()
    # action = agent.compute_single_action(obs)
    # obs, reward, terminated, truncated, _ = env.step(action)
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.compute_single_action(obs)
        action = np.round(discretize_actions(action), 2)
        obs, reward, terminated, truncated, _ = env.step(action)
        sa_pair.append([list(obs), list(action)])
    print(sa_pair)
    episodes_dict[f"episode{ep_counter}"] = sa_pair
    sa_pair = []

print(episodes_dict)

json_object = json.dumps(episodes_dict)
with open("acrobot_ds.json", "w") as outfile:
    outfile.write(json_object)
ray.shutdown()
