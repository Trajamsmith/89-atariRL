import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size: int, input_shape):
        """
        Replay buffers are an essential component of DQNs.
        Without them, our networks wouldn't converge during training.
        Args:
            max_size: Maximum number of actions stored in buffer.
            input_shape: Shape of the state values.
        """
        self.mem_size = max_size

        # Store the current working index
        self.mem_cntr = 0

        # Initialize memory for our five value types
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, 4), dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state: int, action: int,
                         reward: float, state_: int, done: bool) -> None:
        """
        Store the results of an action taken by our agent.
        """
        # Overwrite older array values when exceeding mem_size
        index = self.mem_cntr % self.mem_size

        # Get the one-hot encoding of our discrete action
        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1.0

        # Store all values
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)

        self.mem_cntr += 1

    # Sample a subset of the memory
    def sample_buffer(self, batch_size: int):
        """
        Sample a subset of our buffer memory, to use for
        training our DQN.
        Args:
            batch_size: Number of samples to fetch.
        """
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
