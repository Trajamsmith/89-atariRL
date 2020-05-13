from atari.config import *
from atari.game_wrapper import GameWrapper
from atari.model.dqn_architecture import build_q_network
from atari.model.replay_buffer import ReplayBuffer
from atari.model.agent import Agent

import numpy as np
import tensorflow as tf
import time

# TensorBoard writer
writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

# Create environment
game_wrapper = GameWrapper(MAX_NOOP_STEPS)
print("The environment has the following {} actions: {}".format(game_wrapper.env.action_space.n,
                                                                game_wrapper.env.unwrapped.get_action_meanings()))

# TODO: Move this to another module
# Create or load the agent
if LOAD_FROM is None:
    frame_number = 0
    rewards = []
    loss_list = []

    # Build main and target networks
    MAIN_DQN = build_q_network(LEARNING_RATE, input_shape=INPUT_SHAPE)
    TARGET_DQN = build_q_network(input_shape=INPUT_SHAPE)

    replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE)
    agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, input_shape=INPUT_SHAPE,
                  batch_size=BATCH_SIZE)
else:
    # TODO: LOADING IS A LITTLE BROKEN AT THE MOMENTS!
    # Load the agent instead
    print('Loading from', LOAD_FROM)
    meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)

    # Apply information loaded from meta
    frame_number = meta['frame_number']
    rewards = meta['rewards']
    loss_list = meta['loss_list']

    print('Loaded')

# FULL TRAINING LOOP
try:
    # Allows us to write to Tensorboard
    with writer.as_default():
        while frame_number < TOTAL_FRAMES:
            epoch_frame = 0

            # TRAINING EPOCH
            # We evaluate and save our model after a number of
            # epoch, controlled by frame numbers in our config.
            while epoch_frame < FRAMES_BETWEEN_EVAL:
                start_time = time.time()
                game_wrapper.reset()
                life_lost = True
                episode_reward_sum = 0

                # TRAINING EPISODE
                # One episode is one game, after which we update
                # our metrics. If the episode takes longer than anticipated
                # we can shortcircuit (to avoid less valuable training).
                for _ in range(MAX_EPISODE_LENGTH):
                    # Get action
                    action = agent.get_action(frame_number, game_wrapper.state)

                    # Take step
                    processed_frame, reward, terminal, life_lost = game_wrapper.step(action)
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward

                    # Add experience to replay memory
                    agent.add_experience(action=action,
                                         frame=processed_frame[:, :, 0],
                                         reward=reward, clip_reward=CLIP_REWARD,
                                         terminal=life_lost)

                    # Update agent
                    if frame_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                        loss, _ = agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR, frame_number=frame_number,
                                              priority_scale=PRIORITY_SCALE)
                        loss_list.append(loss)

                    # Update target network
                    if frame_number % UPDATE_FREQ == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
                        agent.update_target_network()

                    # Break the loop when the game is over
                    if terminal:
                        terminal = False
                        break

                rewards.append(episode_reward_sum)

                # Output the progress every 10 games
                if len(rewards) % 10 == 0:
                    # Write to TensorBoard
                    if WRITE_TENSORBOARD:
                        tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                        tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                        writer.flush()

                    print(
                        f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  Average reward: {np.mean(rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')

            # Evaluation every `FRAMES_BETWEEN_EVAL` frames
            terminal = True
            eval_rewards = []
            evaluate_frame_number = 0

            for _ in range(EVAL_LENGTH):
                if terminal:
                    game_wrapper.reset(evaluation=True)
                    life_lost = True
                    episode_reward_sum = 0
                    terminal = False

                # Breakout requires a "fire" action (action #1) to start the
                # game each time a life is lost.
                # Otherwise, the agent would sit around doing nothing.
                action = 1 if life_lost else agent.get_action(frame_number, game_wrapper.state, evaluation=True)

                # Step action
                _, reward, terminal, life_lost = game_wrapper.step(action)
                evaluate_frame_number += 1
                episode_reward_sum += reward

                # On game-over
                if terminal:
                    eval_rewards.append(episode_reward_sum)

            # Examine evaluation scores
            if len(eval_rewards) > 0:
                final_score = np.mean(eval_rewards)
            else:
                # In case the game is longer than the number of frames allowed
                final_score = episode_reward_sum
            # Print score and write to Tensorboard
            print('Evaluation score:', final_score)
            if WRITE_TENSORBOARD:
                tf.summary.scalar('Evaluation score', final_score, frame_number)
                writer.flush()

            # Save model
            print("LENGTH OF REWARDS", len(rewards))
            if len(rewards) > 100 and SAVE_PATH is not None:
                print('SAVING MODEL')
                # Temp store last_checkpoint dir
                prev_dir = agent.last_checkpoint

                # Update location of last checkpoint
                dir_name = f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}'
                agent.last_checkpoint = dir_name

                # Save new checkpoint
                agent.save(dir_name, frame_number=frame_number, rewards=rewards,
                           loss_list=loss_list)

                # Remove old checkpoint to prevent bloat
                # agent.delete_prev_checkpoint(prev_dir)

except KeyboardInterrupt:
    print('\nTraining exited early.')
    writer.close()

    if SAVE_PATH is None:
        try:
            SAVE_PATH = input(
                'Would you like to save the trained model? \
                If so, type in a save path, otherwise, interrupt with Ctrl + C. ')
        except KeyboardInterrupt:
            print('\nExiting...')

    if SAVE_PATH is not None:
        print('Saving...')
        agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards,
                   loss_list=loss_list)
        print('Saved.')
