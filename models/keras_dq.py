from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        # How deep into the past do we want to remember things
        self.mem_size = max_size
        self.mem_cntr = 0

        # Is our space continuous or discrete?
        self.discrete = discrete

        # Store states from the environment
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))

        self.dtype = np.int8 if self.discrete else np.float32

        # Other memory arrays
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        # Ovewrite older array values when exceeding mem_size
        index = self.mem_cntr % self.mem_size

        # Store all values
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)

        if self.discrete:
            # Get the one-hot encoding
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            # For continous spaces, store the float
            self.action_memory[index] = action

        self.mem_cntr += 1

    # Sample a subset of the memory
    def sample_buffer(self, batch_size: int):
        max_mem = min(self.mem_cntr, self.mem_size)

        # Choose <batch_size> indices from 0 to <max_mem>
        batch = np.random.choice(max_mem, batch_size)
        


rb = ReplayBuffer(5, 2, 2)
rb.store_transition(1, 1, 1, 1, 0)
rb.store_transition(1, 1, 1, 1, 0)
rb.store_transition(1, 1, 1, 1, 0)
rb.store_transition(1, 1, 1, 1, 0)
rb.store_transition(1, 1, 1, 1, 0)
rb.sample_buffer(2)
