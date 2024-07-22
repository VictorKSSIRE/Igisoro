import os
import copy
import numpy as np
import random
import time
from keras.models import load_model
from selfplay import SelfPlay
from igisoro import Igisoro
from mcts import MCTS, MCTSNode, AlphaZeroMCTS
from network import PolicyNetwork, ValueNetwork


def evaluate_model(game, policy_network, value_network, num_games=50):
    wins = 0
    total_moves = 0
    total_time = 0
    for _ in range(num_games):
        state = game.get_initial_state()
        move_count = 0
        start_time = time.time()
        #breakpoint()
        while not state.is_terminal():
            if state.current_player == 0:
                move = select_move_model(state, policy_network, value_network)
            else:
                move = baseline_player(state)
            state = state.sow(move)
            move_count += 1

        end_time = time.time()
        total_time += (end_time - start_time)
        total_moves += move_count
        if state.get_reward() > 0:  # Assuming positive reward is a win for the trained model
            wins += 1
    average_moves = total_moves / num_games
    average_time = total_time / num_games

    return wins / num_games, average_moves, average_time


def baseline_player(state):
    #pure_mcts = MCTS(game, n_simulations=50)
    #state_copy = copy.deepcopy(state)
    #best_move = pure_mcts.search(state_copy)
    #return best_move
    legal_moves = state.get_legal_moves()
    return random.choice(legal_moves)


def select_move_model(state, policy_network, value_network):
    #mcts = AlphaZeroMCTS(game, policy_network, value_network, n_simulations=10)
    #state_copy = copy.deepcopy(state)
    #best_move = mcts.search(state_copy)
    #return best_move
    move_probs = policy_network.predict((state.board.reshape(1, 4, 8), np.array([state.current_player]).reshape(1, 1)))
    legal_moves = state.get_legal_moves()
    move_probs = [move_probs[0][convert_move_to_index(move)] for move in legal_moves]
    move = random.choices(legal_moves, weights=move_probs, k=1)[0]
    return move


def convert_move_to_index(move):
    row, col = move
    return row * 8 + col  # Assuming an 8x8 board


if __name__ == "__main__":
    game = Igisoro()
    best_win_rate = 0.0
    best_policy_model_path = ''
    best_value_model_path = ''
    for epoch in range(1):
        print(f"Start")
        policy_model_path = f'models/policy_network_epoch_updated.keras'
        value_model_path = f'models/value_network_epoch_updated.keras'

        # Evaluate the model
        win_rate, average_moves, average_time = evaluate_model(game, load_model(policy_model_path), load_model(value_model_path))
        print(f"Epoch {epoch + 1}: Win Rate: {win_rate * 100:.2f}%, Average Moves: {average_moves:.2f}, Average Time: {average_time:.2f} seconds")

        # Track the best model
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_policy_model_path = policy_model_path
            best_value_model_path = value_model_path

    print(f"Best model saved with win rate: {best_win_rate * 100:.2f}%")
    print(f"Best policy model path: {best_policy_model_path}")
    print(f"Best value model path: {best_value_model_path}")
