from lunar_lander.basic_deep_q import Agent
import numpy as np
import gym

"""
This file trains a basic deep Q-learning model to play the
lunar lander game in gym. The model used does not make use of all
the optimization techniques needed for more complex games (like
the Atari games) but is a good introduction to deep Q-learning.
"""
if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    # Number of episodes (one episode is one game)
    n_games = 500

    # Initialize the agent
    # TODO: Explore more adjustments in these hyperparameters
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=8,
                  n_actions=4, mem_size=1000000, batch_size=64, epsilon_min=0.01)

    # Load model here, if continuing to train an existing model
    agent.load_model()

    # Keep a history of scores and episodes
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0

        # Reset the environment, since we're starting
        # a new game
        observation = env.reset()

        while not done:
            # Render the scene -- COMMENT OUT IF TRAINING
            env.render()

            # Take an action and observe the results
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_

            # Update our network based on the results
            # agent.learn()

        # Store episode results
        eps_history.append(agent.epsilon)
        scores.append(score)

        # Update the exploration rate
        agent.epsilon = agent.epsilon * agent.epsilon_dec if agent.epsilon > agent.epsilon_min \
            else agent.epsilon_min

        avg_score = np.mean(scores[max(0, i - 100):(i + 1)])
        print('Episode: ', i,
              ' / Epsilon: %.3f' % agent.epsilon,
              ' / Score: %.2f' % score,
              ' / Average score: %.2f' % avg_score)

        # Save model periodically
        if i % 10 == 0 and i > 0:
            print('Saving model!')
            agent.save_model()
