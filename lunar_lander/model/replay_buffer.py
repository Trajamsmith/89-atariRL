import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size: int, input_shape,
                 n_actions: int, discrete=False):
        # How deep into the past do we want to remember things
        self.mem_size = max_size
        self.mem_cntr = 0

        # Is our space continuous or discrete?
        self.discrete = discrete

        # Store states from the environment
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))

        # Other memory arrays
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state: int, action: int,
                         reward: float, state_: int, done: bool) -> None:
        # Overwrite older array values when exceeding mem_size
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
            # For continuous spaces, store the float
            self.action_memory[index] = action

        self.mem_cntr += 1

    # Sample a subset of the memory
    def sample_buffer(self, batch_size: int):
        max_mem = min(self.mem_cntr, self.mem_size)

        # Choose <batch_size> indices from 0 to <max_mem>
        batch = np.random.choice(max_mem, batch_size)

        # Sample SARS across the buffer
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
