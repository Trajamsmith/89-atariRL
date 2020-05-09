from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
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

        # Sample SARS across the buffer
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


rb = ReplayBuffer(5, 2, 2)
rb.store_transition(1, 1, 1, 1, 0)
rb.store_transition(1, 1, 1, 1, 0)
rb.store_transition(1, 1, 1, 1, 0)
rb.store_transition(1, 1, 1, 1, 0)
rb.store_transition(1, 1, 1, 1, 0)
rb.sample_buffer(2)


def build_dqn(lr: float, n_actions: int,
              input_dims, fc1_dims, fc2_dims) -> Sequential:
    """
    Build our deep-Q network. Note that depending on the problem space,
    the deep Q network could very well be a multi-layer dense network,
    a convolutional network, or even a recurrent network. Here we're
    using a multi-layer, fully connected network.
    Args:
        lr: The learning rate.
        n_actions: The number of actions the agent can take.
        input_dims: The input dimension to our NN.
        fc1_dims: The first fully-connected layer's dimensions.
        fc2_dims: The second fully-connected layer's dimensions.
    Returns: A compiled Keras model.
    """
    model = Sequential([
        # Leaving (input_dims,) allows us to pass in either a
        # single memory, or a batch of memories.
        Dense(fc1_dims, input_shape=(input_dims,)),
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        Dense(n_actions)
    ])

    model.compile(optimizer=Adam(lr=lr), loss='mse')
    return model


class Agent(object):
    """
    Args:
        alpha: The learning rate.
        gamma: The discount factor.
        n_actions: Number of actions.
        epsilon: The exploration-exploitation parameter.
        batch_size: Size of the training sample.
        input_dims: Input to NN.
        epsilon_dec: Decrement factor for epsilon.
        epsilon_min: Minimum value of epsilon.
        mem_size: Size of our replay buffer.
        fname: Name of the file where we're storing the model.
    """

    def __init__(self, alpha: float, gamma: float, n_actions: int, epsilon: float, batch_size: int,
                 input_dims, epsilon_dec: float = 0.996, epsilon_min: float = 0.01,
                 mem_size: int = 1000000, fname='dqn_model.h5'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = fname

        # Available actions at any given time
        # e.g. [0, 1, 2, 3, 4]
        self.action_space = [i for i in range(n_actions)]

        # Init our buffer
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        # Init our neural network
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

    # Store system snapshot in buffer
    def remember(self, state: int, action: int, reward: float,
                 new_state: int, done: bool) -> None:
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state) -> int:
        # Adds an axis to our vector so we can feed it
        # to our neural network, regardless of shape.
        state = state[np.newaxis, :]

        rand = np.random.random()
        if rand < self.epsilon:
            # Explore - Choose action at random.
            action = np.random.choice(self.action_space)
        else:
            # Exploit - Choose the best action based
            # on our learned action-value function.
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    # Temporal difference learning process.
    def learn(self) -> None:
        # Two important breakthroughs that make this work in
        # situations where naive Q didn't --
        # 1. EXPERIENCE REPLAY
        # 2. SEPARATELY UPDATED TARGET NETWORK

        if self.memory.mem_cntr < self.batch_size:
            # We need to fill our memory before we can
            # start the learning process. We're choosing
            # to keep a buffer full of zeros, instead
            # of randomly initializing the arrays.
            return

        # Note that we're generally sampling non-sequential memories
        # TODO: So, are we just jumping around, sampling random states from
        # TODO: different parts of the games? Still a little lost.
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # Working with the one-hot encoding
        action_values = np.array(self.action_space, dtype=np.int8)
        # Dot product, e.g. [1, 2, 3, 4] with [0, 0, 1, 0] for
        # integer values from one-hot encodings
        action_indices = np.dot(action, action_values)

        # Predict current reward
        q_eval = self.q_eval.predict(state)
        # Predict reward at step n + 1
        q_next = self.q_eval.predict(new_state)

        # We have to simultaneously train our target network!
        # This is the only way we can effectively evaluate a loss
        # function and calculate our gradient descent.
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Here's our version of the Bellman equation
        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done

        # Update our parameters: q_target is our validation
        _ = self.q_eval.fit(x=state, y=q_target, verbose=0)

        # Update epsilon
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min \
            else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
