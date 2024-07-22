import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Concatenate

# Basic


class PolicyNetwork:
    def __init__(self, num_actions=32):
        board_input = Input(shape=(4, 8), name='board_input')
        player_input = Input(shape=(1,), name='player_input')

        x = Flatten()(board_input)
        x = Concatenate()([x, player_input])
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        policy_output = Dense(num_actions, activation='softmax', name='policy_output')(x)

        self.model = Model(inputs=[board_input, player_input], outputs=policy_output)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self, states, moves):
        boards, players = states
        boards = np.array(boards)
        players = np.array(players)
        self.model.fit([boards, players], moves, epochs=10, batch_size=32)

    def predict(self, state):
        board, player = state
        board = np.array(board).reshape(1, 4, 8)  # Add batch dimension
        player = np.array(player).reshape(1, 1)  # Add batch dimension
        return self.model.predict([board, player])


class ValueNetwork:
    def __init__(self):
        board_input = Input(shape=(4, 8), name='board_input')
        player_input = Input(shape=(1,), name='player_input')

        x = Flatten()(board_input)
        x = Concatenate()([x, player_input])
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        value_output = Dense(1, activation='linear', name='value_output')(x)

        self.model = Model(inputs=[board_input, player_input], outputs=value_output)
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def fit(self, states, rewards):
        boards, players = states
        boards = np.array(boards)
        players = np.array(players)
        self.model.fit([boards, players], rewards, epochs=10, batch_size=32)

    def predict(self, state):
        board, player = state
        board = np.array(board).reshape(-1, 4, 8)  # Add batch dimension
        player = np.array(player).reshape(-1, 1)  # Add batch dimension
        return self.model.predict([board, player])

# AlphaGo Zero


def residual_block(x, filters, kernel_size=3):
    """
    A residual block with two convolutional layers.
    """
    shortcut = x

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def build_policy_value_network(board_input_shape=(4, 8, 1), num_actions=32):
    """
    Builds a neural network that outputs both policy and value.
    """
    board_input = Input(shape=board_input_shape, name='board_input')
    player_input = Input(shape=(1,), name='player_input')

    x = Conv2D(64, (3, 3), padding='same')(board_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Add residual blocks
    for _ in range(5):  # Number of residual blocks
        x = residual_block(x, 64)

    # Flatten the convolutional output
    x = Flatten()(x)

    # Concatenate with the player input
    x = Concatenate()([x, player_input])

    # Policy head
    policy_x = Dense(256, activation='relu')(x)
    policy_output = Dense(num_actions, activation='softmax', name='policy_output')(policy_x)

    # Value head
    value_x = Dense(256, activation='relu')(x)
    value_output = Dense(1, activation='tanh', name='value_output')(value_x)

    model = Model(inputs=[board_input, player_input], outputs=[policy_output, value_output])
    model.compile(optimizer='adam',
                  loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mean_squared_error'})
    return model


class PolicyValueNetwork:
    def __init__(self, board_input_shape=(4, 8, 1), num_actions=32):
        self.model = build_policy_value_network(board_input_shape, num_actions)

    def fit(self, states, policy_targets, value_targets, epochs=10, batch_size=32):
        boards, players = states
        boards = np.array(boards).reshape(-1, 4, 8, 1)  # Adjust shape as needed
        players = np.array(players).reshape(-1, 1)  # Ensure players are shaped correctly
        self.model.fit([boards, players], {'policy_output': policy_targets, 'value_output': value_targets}, epochs=epochs, batch_size=batch_size)

    def predict(self, state):
        board, player = state
        board = np.array(board).reshape(-1, 4, 8, 1)  # Add batch dimension
        player = np.array(player).reshape(-1, 1)  # Add batch dimension
        # Debugging print statements
        print(f"Board shape: {board.shape}")
        print(f"Player shape: {player.shape}")
        return self.model.predict([board, player])
