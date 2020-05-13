import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from lunar_lander.model.replay_buffer import ReplayBuffer
from lunar_lander.model.dqn import build_dqn


class Agent(object):
    def __init__(self, alpha: float, gamma: float, epsilon: float, batch_size: int,
                 input_dims: int, epsilon_dec: float = 0.998, epsilon_min: float = 0.01,
                 mem_size: int = 1000000, fname: str = 'll_model.h5'):
        """
        Our Lunar Lander agent, using a pretty straightforward
        deep Q network with experience replay.
        Args:
            alpha: The learning rate.
            gamma: The discount factor.
            epsilon: The exploration-exploitation parameter.
            batch_size: Size of the training sample.
            input_dims: Input to NN.
            epsilon_dec: Decrement factor for epsilon.
            epsilon_min: Minimum value of epsilon.
            mem_size: Size of our replay buffer.
            fname: Name of the file where we're storing the model.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = fname

        # Available actions at any given time
        # e.g. [0, 1, 2, 3, 4]
        self.action_space = [i for i in range(4)]

        # Init our buffer
        self.memory = ReplayBuffer(mem_size, input_dims)

        # Init our neural network
        # From a paper I found, two dense nets with 256 and 128 size is best
        self.q_eval = build_dqn(alpha, input_dims, 256, 128)

        # Tensorboard logs
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/ll/' + current_time
        self.summary_writer = tf.summary.create_file_writer(log_dir)

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
    def episode(self):
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
        history = self.q_eval.fit(x=state, y=q_target, verbose=0)
        return history

    def update_eps(self):
        """
        Update epsilon after an episode
        """
        self.epsilon = (self.epsilon * self.epsilon_dec) if self.epsilon > self.epsilon_min \
            else self.epsilon_min

    def log_metrics(self, episode: int, reward: float, avg_rewards: float, avg_losses: float):
        """
        Log our metrics so we can visualize them in TensorBoard
        """
        with self.summary_writer.as_default():
            tf.summary.scalar('episode reward', reward, step=episode)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=episode)
            tf.summary.scalar('average loss', avg_losses, step=episode)

    def save_model(self):
        print("Model saved!")
        self.q_eval.save(self.model_file)

    def load_model(self):
        print("Model loaded!")
        self.q_eval = load_model(self.model_file)
