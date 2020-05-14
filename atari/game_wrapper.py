import random
import gym
import cv2
import numpy as np
from typing import Literal


def process_frame(frame, shape=(84, 84)):
    """
    Using full-scale, RBG images for our network training
    is just too computationally demanding. We can reduce
    them to 84x84 grayscale images.
    Args:
        frame: The frame to process. Must have values ranging from 0-255.
        shape: The desired output shape.
    Returns:
        The processed frame
    """
    frame = frame.astype(
        np.uint8)  # cv2 requires np.uint8, other dtypes will not work

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34 + 160, :160]  # crop image
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))

    return frame


class GameWrapper:
    def __init__(self, no_op_steps: int = 10, history_length: int = 4):
        """
        Wrapper for the environment provided by Gym.
        Args:
            no_op_steps: Evaluation "lag," improve performance
            history_length: History length
        """
        self.env = gym.make('BreakoutDeterministic-v4')
        self.no_op_steps = no_op_steps
        self.history_length = history_length

        self.state = None
        self.last_lives = 0

    def reset(self, evaluation: bool = False):
        """
        Fully reset the environment, and set the initial state to the initial frame repeated 4 times.
        Args:
            evaluation: Set to True when the agent is being evaluated. Takes a random number of no-op steps if True.
        """
        self.frame = self.env.reset()
        self.last_lives = 0

        # If evaluating, take a random number of no-op steps.
        # This adds an element of randomness, so that the each
        # evaluation is slightly different.
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)

        # For the initial state, we stack the first frame four times
        self.state = np.repeat(process_frame(self.frame),
                               self.history_length, axis=2)

    def step(self, action: int, render_mode: Literal['human', 'rgb_array'] = None):
        """
        Performs an action and observes the result
        Args:
            action: An integer describe action the agent chose
            render_mode: None doesn't render anything, 'human' renders the screen in a new window, 'rgb_array' returns  an np.array with rgb values
        Returns:
            processed_frame: The processed new frame as a result of that action
            reward: The reward for taking that action
            terminal: Whether the game has ended
            life_lost: Whether a life has been lost
            new_frame: The raw new frame as a result of that action. If render_mode is set to 'rgb_array' this also returns the rendered rgb_array
        """
        new_frame, reward, terminal, info = self.env.step(action)

        # In the commonly ignored 'info' or 'meta' data returned by env.step
        # we can get information such as the number of lives the agent has.

        # We use this here to find out when the agent loses a life, and
        # if so, we set life_lost to True.

        # We use life_lost to force the agent to start the game
        # and not sit around doing nothing.
        if info['ale.lives'] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info['ale.lives']

        processed_frame = process_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)

        if render_mode == 'rgb_array':
            return processed_frame, reward, terminal, life_lost, self.env.render(render_mode)
        elif render_mode == 'human':
            self.env.render()

        return processed_frame, reward, terminal, life_lost
