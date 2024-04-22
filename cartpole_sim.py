import gymnasium as gym
import numpy as np
import torch
from model import MultiHeadedCNN
import time


def run_cartpole_env():
    STATE_LENGTH = 6
    CARTPOLE_CLASSES_MAP = {0: -10.0,
                            1: 0,
                            2: 1}

    model = MultiHeadedCNN(num_classes=[4, 3, 4, 6])
    print("Weights before loading: ", model.conv_layers[0].weight[0][0][0][0])
    model.load_state_dict(torch.load("trained_models/MultiHeaded-CNN-3-1.pth", map_location=torch.device("cpu")))
    print("Weights after loading: ", model.conv_layers[0].weight[0][0][0][0])

    model.eval()

    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset()
    state_space_dimension = obs.shape[0]
    terminated, truncated = False, False
    t = 0
    state_history = []
    actions_log = []
    action = env.action_space.sample()  # initial random action

    while not (terminated or truncated):
        time.sleep(0.05)
        print(f"action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)

        state_history.append(obs.tolist())
        # print("obs type:", type(obs.tolist()))
        # print("obs :", obs.tolist())
        if t >= 6:
            state_0 = state_history[t - 6]
            state_1 = state_history[t - 5]
            state_2 = state_history[t - 4]
            state_3 = state_history[t - 3]
            state_4 = state_history[t - 2]
            state_5 = state_history[t - 1]
            state_6 = state_history[t]
            print(type(state_0))
            print(state_0)

            seven_states = [state_0] + [state_1] + [state_2] + [state_3] + [state_4] + [state_5] + [state_6]
            print("before conversion:", seven_states)

            padded_states = []

            if state_space_dimension != STATE_LENGTH:
                for state in seven_states:
                    if len(state) != STATE_LENGTH:
                        additional_elements = STATE_LENGTH - state_space_dimension
                        state.extend([-100] * additional_elements)
                print("seven_states", seven_states)
                # convert the sublists into ndarrays
                padded_states = [np.array(arr) for arr in seven_states]
                # convert the list into ndarray
                padded_states = np.array(padded_states)
            else:
                padded_states = [np.array(arr) for arr in seven_states]
                padded_states = np.array(padded_states, dtype=np.float32)

            print("after conversion, before transposing:", padded_states)
            # print("after conversion:", padded_states.T)
            env_states = padded_states.T
            print("after conversion, after transposing:", env_states)
            env_states = np.reshape(env_states, (1, 6, 7))
            print("after conversion, after transposing, after reshaping:", env_states)
            dummy_states_top = np.full((1, 6, 7), -100.0)
            dummy_states_bottom = np.full((2, 6, 7), -100.0)
            input_states = np.concatenate((dummy_states_top, env_states, dummy_states_bottom), axis=0)
            # print(f"manipulated input_data:{input_states}")
            input_states = torch.from_numpy(input_states).float()
            input_states = input_states.view(-1, 4, 6, 7)
            print("input states ", input_states)
            print("input states shape", input_states.shape)

            with torch.no_grad():
                output1, output2, output3, output4 = model(input_states)
            print(f"output1: {output1}")
            print(f"output2: {output2}")
            print(f"output3: {output3}")
            print(f"output4: {output4}")
            action_prob = torch.nn.functional.softmax(output2, dim=1)
            print("actions probabilities:", action_prob)
            _, predicted_class = torch.max(action_prob, 1)
            print("predicted class:", predicted_class.item())
            predicted_action = CARTPOLE_CLASSES_MAP[predicted_class.item()]

            if predicted_action == -10.0:
                # error handling the dummy action output, since can't pass -10.0 so instead pass 1 is like nothing in Acrobot
                action = env.action_space.sample()
            else:
                action = predicted_action

        else:
            print("Im triggered")
            action = env.action_space.sample()

        t += 1

    env.close()
