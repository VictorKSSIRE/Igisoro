import os
import numpy as np
from igisoro import Igisoro
from network import PolicyNetwork, ValueNetwork


def select_move(state, policy_network):
    # Use the policy network to predict the next move
    state_array = np.array([state.board])
    move_probabilities = policy_network.model.predict(state_array)[0]
    legal_moves = state.get_legal_moves()

    # Filter the move probabilities to only include legal moves
    legal_move_probabilities = np.zeros_like(move_probabilities)
    for move in legal_moves:
        row, col = move
        legal_move_probabilities[row * 8 + col] = move_probabilities[row * 8 + col]

    # Normalize the probabilities
    if np.sum(legal_move_probabilities) > 0:
        legal_move_probabilities /= np.sum(legal_move_probabilities)

    # Select a move based on the probabilities
    flat_move = np.random.choice(range(len(legal_move_probabilities)), p=legal_move_probabilities)
    move = (flat_move // 8, flat_move % 8)
    return move


def play_against_model():
    game = Igisoro()

    # Load the trained models
    policy_network = PolicyNetwork()
    value_network = ValueNetwork()
    policy_network.model.load_weights('models/policy_network_epoch_6.keras')  # Load the last epoch or desired epoch
    value_network.model.load_weights('models/value_network_epoch_6.keras')

    print("Let's play Igisoro! You are Player 1.")
    game.display_board()

    while not game.is_terminal():
        if game.current_player == 1:
            # Human player's turn
            while True:
                try:
                    position_input = input("Your move (row, col): ")
                    position = tuple(map(int, position_input.strip("()").split(",")))
                    if game.valid_move(position[0], position[1]):
                        game.sow(position)
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input format. Please enter a tuple (row, col).")
        else:
            # Model's turn
            state = game.get_curr_state()
            move = select_move(state, policy_network)
            game.sow(move)
            print(f"Model move: {move}")

        #game.current_player = 1 - game.current_player
        game.display_board()

    reward = game.get_reward()
    if reward > 0:
        print("Model wins!")
    elif reward < 0:
        print("You win!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    play_against_model()
