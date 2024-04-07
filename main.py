import gymnasium as gym
import numpy as np
from multiprocessing import Process
import time
import torch
from model import MultiHeadedCNN


ACROBOT_CLASSES_MAP = {0: -10.0, 1: 0, 2: 1, 3: 2}

model = MultiHeadedCNN(num_classes=[4, 3, 4, 6])
model.load_state_dict(torch.load("trained_models/MultiHeaded-CNN-1-2.pth", map_location=torch.device("cpu")))
model.eval()

env = gym.make("Acrobot-v1", render_mode="human")
obs, info = env.reset()
terminated, truncated = False, False
t = 0
state_history = []
actions_log = []
action = env.action_space.sample()  # initial random action

while not (terminated or truncated):
   print(f"acrtion:{action}")
   obs, reward, terminated, truncated, info = env.step(action)

   state_history.append([obs])
   if t >= 7:
      state_0 = state_history[t-6]
      state_1 = state_history[t-5]
      state_2 = state_history[t-4]
      state_3 = state_history[t-3]
      state_4 = state_history[t-2]
      state_5 = state_history[t-1]
      state_6 = state_history[t]

      seven_states = [state_0 + state_1 + state_2 + state_3
                      + state_4 + state_5 + state_6]

      np_seven_states = np.array(seven_states)
      env_states = np.array([arr.T for arr in np_seven_states], dtype=np.float32)
      dummy_states = np.full((3, 6, 7), -100)
      input_states = np.concatenate((env_states, dummy_states), axis=0)
      input_states = torch.from_numpy(input_states).float()
      input_states = input_states.view(-1, 4, 6, 7)

      with torch.no_grad():
         output1, output2, output3, output4 = model(input_states)
      action_prob = torch.nn.functional.softmax(output1, dim=1)
      print("actions probabilities:", action_prob)
      _, predicted_class = torch.max(action_prob, 1)
      predicted_action = ACROBOT_CLASSES_MAP[predicted_class.item()]

      if predicted_action == -10.0:
         # error handling the dummy action output, since can't pass -10.0 so instead pass 1 is like nothing in Acrobot
         action = 1

      action = predicted_action

   else:
      action = env.action_space.sample()
   t += 1

env.close()

# def run_environment(env_name):
#    env = gym.make(env_name, render_mode="human")
#    obs, info = env.reset()
#    for _ in range(100):
#       time.sleep(0.1)
#       action = env.action_space.sample()
#       obs, reward, terminated, truncated, info = env.step(action)
#    env.close()
#
# if __name__ == "__main__":
#    environments = ["Acrobot-v1", "CartPole-v1"]
#
#    processes = [Process(target=run_environment, args=(env_name,)) for env_name in environments]
#
#    for process in processes:
#       process.start()
#
#    for process in processes:
#       process.join()