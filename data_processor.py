import json

import numpy as np


class DataProcessor:
    state_space_dimension = 6

    def __init__(self):
        self.__all_states = []
        self.__all_actions = []

    @property
    def all_states(self):
        return self.__all_states

    @property
    def all_actions(self):
        return self.__all_actions

    @staticmethod
    def load_data_from_json(file_path):
        """Load dictionary of episodes from json file"""
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def process_data(data):
        """given the dictionary of episodes, concatenate the first 7
         states with the last action, and return each separately from the other"""
        temp_s, temp_a = [], []
        parse_dataset_s, parse_dataset_a = [], []
        for i in range(1, len(data) + 1):
            episode = data[f"episode{i}"]
            if len(episode) >= 7:
                # check whether the episode is in the threshold horizon
                for t in range(len(episode) - 7):
                    state_0, action_0 = episode[t]
                    state_1, action_1 = episode[t + 1]
                    state_2, action_2 = episode[t + 2]
                    state_3, action_3 = episode[t + 3]
                    state_4, action_4 = episode[t + 4]
                    state_5, action_5 = episode[t + 5]
                    state_6, action_6 = episode[t + 6]

                    temp_s += [[state_0] + [state_1] + [state_2] + [state_3]
                               + [state_4] + [state_5] + [state_6]]
                    temp_a.append(action_6)

                parse_dataset_s += temp_s
                parse_dataset_a += temp_a
                temp_s, temp_a = [], []  # empty temporary 7 states 1 action pair's list

        return parse_dataset_s, parse_dataset_a

    @staticmethod
    def crop_data(states, actions):
        return states[:820000], actions[:820000]

    def round_actions(self):
        """
        round the actions to 2 decimal points
        convert from list to numpy array
        """
        self.__all_actions = np.round(self.__all_actions, 2)

    def convert_states_to_np(self):
        """convert the states to a numpy array"""
        self.__all_states = np.array(self.__all_states)

    def combine_data(self, states, actions):
        self.__all_states.extend(states)
        self.__all_actions.extend(actions)

    def process_file(self, file_path):
        data = self.load_data_from_json(file_path)
        states, actions = self.process_data(data)
        states, actions = DataProcessor.crop_data(states, actions)
        self.combine_data(states, actions)

    def add_dummy_states(self):
        """
        Ensures each state in each episode within self.__all_states has a length equal to state_space_dimension.
        This is achieved by appending zeros to states shorter than state_space_dimension.
        """
        # Check whether there is any need for the dummy states addition
        if self.__all_states[0][0] != self.state_space_dimension:
            # Iterate through each 7 states collection in self.__all_states
            for seven_states in self.__all_states:
                # Iterate through each state in the current episode
                for one_state in seven_states:
                    # Calculate the number of zeros needed to make the
                    # state's length equal to state_space_dimension
                    additional_elements = self.state_space_dimension - len(one_state)
                    one_state.extend([0] * additional_elements)  # This modifies the 'state' in-place,
                    # which is a part of self.__all_states

    def reshaped_to_2d_np_array(self):
        """
        Take the transpose after converting them to np arrays, so that
        on the rows we have the state and on the columns we have states
        with respect to t horizon
        """
        self.convert_states_to_np()
        self.round_actions()
        self.__all_states = np.array([arr.T for arr in self.__all_states], dtype=np.float32)


if __name__ == "__main__":
    acrobot_env_path = "../ENV_datasets/acrobot_ds.json"
    acrobot_env = DataProcessor()
    acrobot_env.process_file(acrobot_env_path)
    acrobot_env.add_dummy_states()
    acrobot_env.reshaped_to_2d_np_array()

    print(acrobot_env.all_states.shape)
