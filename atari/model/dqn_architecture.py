from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf


def build_q_network(learning_rate: float = 0.00001,
                    input_shape: tuple = (84, 84), history_length: int = 4) -> Model:
    """
    Builds a dueling DQN as a Keras model. For a good overview of dueling
    DQNs (and some motivation behind their use) see:
    https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751
    Arg:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
        history_length: Number of historical frames the agent can see
    Returns:
        A compiled Keras model
    """
    # Dueling architecture requires a non-sequential step at the end, so
    # Keras's functional API is a natural choice
    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    x = Lambda(lambda layer: layer / 255)(model_input)  # normalize by 255

    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)
    x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)

    # Split into value and advantage streams
    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)  # custom splitting layer

    # State value estimator
    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

    # Advantage value estimator
    # Each of the four actions has its own advantage value
    adv_stream = Flatten()(adv_stream)
    adv = Dense(4, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

    # Combine streams into Q-Values
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))
    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model
