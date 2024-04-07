import numpy as np


def discretize_actions(action, bins_num=6, min_val=-2.0, max_val=2.0):
    """Discretize action space"""
    intervals = np.linspace(min_val, max_val, bins_num)

    # digitize, returns the index of action in interval
    return intervals[np.digitize(action, intervals) - 1]


def create_combined_dataset(environments):
    num_samples = 200000  # number of each environment's instances
                          # (MUST be equal to that of the crop method @data_processor class)

    if len(environments) != 4:
        raise ValueError("The environments argument must contain exactly 4 environments.")

    # Prepare dummy data
    dummy_state = np.full((num_samples, 6, 7), -100)
    dummy_action = np.full((num_samples,), -10)

    # Prepare real data from environments
    real_data = [(env[0], env[1]) for env in environments]  # List of tuples (states, actions) for each environment

    # Define combinations based on binary strings (basically to create a set of generalization)
    combinations = ["{0:04b}".format(i) for i in range(16)]  # Generates '0000' to '1111'

    # Initialize lists for combined data
    combined_all_states = []
    combined_all_actions = [[] for _ in range(4)]  # One sublist per environment action (since we'll have
                                                   # multihead output network corresponding to each env)

    # Generate combined datasets for each combination
    for combo in combinations:
        combined_states = [real_data[i][0] if bit == '1' else dummy_state for i, bit in enumerate(combo)]
        # When setting up combined_actions in the dataset generation loop:
        combined_actions = [(real_data[i][1].flatten() if bit == '1' else dummy_action) for i, bit in enumerate(combo)]


        # Stack states along a new axis to form 3D inputs for each combination
        combined_states = np.stack(combined_states, axis=1)  # Shape will be (num_samples, 4, 6, 7)
        combined_all_states.append(combined_states)

        # Actions are kept separate for each environment
        for i, action_set in enumerate(combined_actions):
            combined_all_actions[i].append(action_set)

    # Final concatenation to merge different combinations into a single dataset
    combined_all_states = np.concatenate(combined_all_states, axis=0)  # Final shape: (16*num_samples, 4, 6, 7)

    # Concatenate actions for each environment across all combinations
    for i in range(4):
        combined_all_actions[i] = np.concatenate(combined_all_actions[i], axis=0)  # Final shape for each: (16*num_samples,)

    # Return the combined dataset
    return (combined_all_states, *combined_all_actions)