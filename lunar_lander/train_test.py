import gym
import numpy as np

from lunar_lander.model.agent import Agent

# If false, load the saved model instead
TRAIN_MODEL: bool = False

"""
This file trains a basic deep Q-learning model to play the
lunar lander game in gym. The model used does not make use of all
the optimization techniques needed for more complex games (like
the Atari games) but is a good introduction to deep Q-learning.
"""
if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    # Number of episodes (one episode is one game)
    n_games = 2000

    # Initialize the agent
    # TODO: Explore more adjustments in these hyperparameters
    epsilon = 1.0
    if not TRAIN_MODEL:
        # We don't want the agent to explore if we're not training
        epsilon = 0.01
    agent = Agent(gamma=0.99, epsilon=epsilon, alpha=0.0001, input_dims=8,
                  n_actions=4, mem_size=1000000, batch_size=64, epsilon_min=0.01)

    # Load model here, if continuing to train an existing model
    if not TRAIN_MODEL:
        agent.load_model()

    # Keep some metric histories
    scores = []
    eps_history = []
    losses = []

    for i in range(n_games):
        done = False
        score = 0
        episode_losses = []

        # Reset the environment, since we're starting
        # a new game
        observation = env.reset()

        while not done:
            # Render the scene
            if not TRAIN_MODEL:
                env.render()

            # Take an action and observe the results
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_

            # Update our network based on the results
            if TRAIN_MODEL:
                history = agent.episode()
                if history:
                    loss = history.history['loss'][0]
                    episode_losses.append(loss)

        # Store the average training loss for this episode
        avg_ep_losses = float(np.mean(episode_losses))
        losses.append(avg_ep_losses)

        # Store other episode results
        eps_history.append(agent.epsilon)
        scores.append(score)

        # Update the exploration rate
        agent.update_eps()

        # Avg score and loss over last ten episodes
        avg_score = float(np.mean(scores[max(0, i - 100):(i + 1)]))
        avg_loss = float(np.mean(losses[max(0, i - 100):(i + 1)]))
        print('Episode: ', i,
              ' / Epsilon: %.3f' % agent.epsilon,
              ' / Score: %.2f' % score,
              ' / Avg. Score: %.2f' % avg_score,
              ' / Loss: %.2f' % losses[-1],
              ' / Avg. Loss: %.2f' % avg_loss)
        agent.log_metrics(episode=i, reward=score, avg_rewards=avg_score, avg_losses=avg_loss)

        # Save model periodically
        if TRAIN_MODEL and i % 10 == 0 and i > 0:
            agent.save_model()
