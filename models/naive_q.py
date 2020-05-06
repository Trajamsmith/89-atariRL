import numpy as np
import gym
from gym.spaces import Discrete, Box
from collections import defaultdict

Q = defaultdict(float)
gamma = 0.99  # Discounting factor
alpha = 0.5  # soft update param

env = gym.make("CartPole-v0")
actions = range(env.action_space)


def update_Q(s, r, a, s_next, done):
    max_q_next = max([Q[s_next, a] for a in actions])
    # Do not include the next state's value if currently at the terminal state.
    Q[s, a] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[s, a])


# -------------------------------------------------------------
# DISCRETIZE THE CONTINUOUS SPACE
# -------------------------------------------------------------
class DiscretizedObservationWrapper(gym.ObservationWrapper):
    """This wrapper converts a Box observation into a single integer.
    """

    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        self.n_bins = n_bins
        self.val_bins = [
            np.linspace(l, h, n_bins + 1) for l, h in zip(low.flatten(), high.flatten())
        ]
        self.observation_space = Discrete(n_bins ** low.flatten().shape[0])

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [
            np.digitize([x], bins)[0]
            for x, bins in zip(observation.flatten(), self.val_bins)
        ]
        return self._convert_to_one_number(digits)


env = DiscretizedObservationWrapper(
    env, n_bins=8, low=[-2.4, -2.0, -0.42, -3.5], high=[2.4, 2.0, 0.42, 3.5]
)

# -------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------
n_steps = 100000
epsilon = 0.1  # 10% chances to apply a random action


def act(ob):
    if np.random() < epsilon:
        # action_space.sample() is a convenient function to get a random action
        # that is compatible with this given action space.
        return env.action_space.sample()

    # Pick the action with highest q value.
    qvals = {a: q[state, a] for a in actions}
    max_q = max(qvals.values())
    # In case multiple actions have the same maximum q value.
    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)


ob = env.reset()
rewards = []
reward = 0.0

for step in range(n_steps):
    a = act(ob)
    ob_next, r, done, _ = env.step(a)
    update_Q(ob, r, a, ob_next, done)
    reward += r
    if done:
        rewards.append(reward)
        reward = 0.0
        ob = env.reset()
    else:
        ob = ob_next
