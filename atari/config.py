# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'BreakoutDeterministic-v4'

# Loading and saving information.
# If LOAD_FROM is None, it will train a new agent.
# If SAVE_PATH is None, it will not save the agent
LOAD_FROM = None
SAVE_PATH = './atari/saved_models'
LOAD_REPLAY_BUFFER = True

WRITE_TENSORBOARD = True
TENSORBOARD_DIR = './atari/logs'

# How much the replay buffer should sample based on priorities.
# 0 = complete random samples, 1 = completely aligned with priorities
PRIORITY_SCALE = 0.7

# Any positive reward is +1, and negative reward is -1, 0 is unchanged
CLIP_REWARD = True

# Total number of frames to train for
TOTAL_FRAMES = 5000000

# Maximum length of an episode (in frames).
# 18000 frames / 60 fps = 5 minutes
MAX_EPISODE_LENGTH = 18000

# Number of frames between evaluations
FRAMES_BETWEEN_EVAL = 250000

# Number of frames to evaluate for
EVAL_LENGTH = 1000

# Number of actions chosen between updating the target network
UPDATE_FREQ = 10000

# Gamma, how much to discount future rewards
DISCOUNT_FACTOR = 0.99

# The minimum size the replay buffer must be
# before we start to update the agent
MIN_REPLAY_BUFFER_SIZE = 50000

# The maximum size of the replay buffer
MEM_SIZE = 1000000

# Randomly perform this number of actions before
# every evaluation to give it an element of randomness
MAX_NOOP_STEPS = 20

# Number of actions between gradient descent steps
UPDATE_FREQ = 4

# Size of the preprocessed input frame. With the
# current model architecture, anything below ~80 won't work.
INPUT_SHAPE = (84, 84)

# Number of samples the agent learns from at once
BATCH_SIZE = 32

LEARNING_RATE = 0.00001