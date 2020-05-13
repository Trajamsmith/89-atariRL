from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_dqn(lr: float, n_actions: int,
              input_dims: int, fc1_dims: int, fc2_dims: int) -> Sequential:
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
