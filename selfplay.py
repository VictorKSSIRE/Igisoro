import numpy as np
from mcts import MCTS, AlphaZeroMCTS
from igisoro import Igisoro
import copy
import json
import os

class SelfPlay:
    def __init__(self, game, policy_network, value_network, n_games, data_file='new_training_data.json'):
        self.game = game
        self.policy_network = policy_network
        self.value_network = value_network
        self.n_games = n_games
        self.select_move_count = 0
        self.data_file = data_file

    def play_games(self):
        training_data = []
        average_moves = 0
        for game_index in range(self.n_games):
            print(f"Playing game {game_index + 1} / {self.n_games}")
            game_data = self.play_single_game()
            print(f"Game {game_index + 1} generated {len(game_data)} moves")
            training_data.extend(game_data)
            average_moves += len(game_data)
        average_moves /= self.n_games
        return training_data, average_moves

    def play_single_game(self):
        #breakpoint()
        state = self.game.get_initial_state()
        game_data = []
        while not state.is_terminal():
            #print(f"Current player: {state.current_player}")
            #print(f"Legal moves: {state.get_legal_moves()}")
            move = self.select_move(state)
            #print(f"Selected move: {move}")
            if move not in state.get_legal_moves():
                #print(f"Move {move} is not legal. Skipping turn.")
                continue
            game_data.append((state.board.copy(), move))
            state = state.sow(move)
        reward = state.get_reward()
        print(f"Game ended with reward: {reward}")
        for i in range(len(game_data)):
            game_data[i] = (game_data[i][0], game_data[i][1], reward)
        return game_data

    def select_move(self, state):
        self.select_move_count += 1
        #print(f"#select_moves : {self.select_move_count}")
        mcts = MCTS(self.game, n_simulations=100)  # should be 1000
        state_copy = copy.deepcopy(state)
        best_move = mcts.search(state_copy)
        return best_move

    def train_policy_value_networks(self, training_data):
        print(f"Training with {len(training_data)} data points")
        if len(training_data) == 0:
            print("No training data generated.")
            return
        states, moves, rewards = zip(*training_data)
        boards, players = zip(*states)
        boards = np.array(boards)
        players = np.array(players)
        moves = np.array(moves)
        moves = np.array([self.convert_move_to_index(move) for move in moves])
        rewards = np.array(rewards)
        print(f"boards.shape[0] = {boards.shape[0]}")
        print(f"players.shape[0] = {players.shape[0]}")
        print(f"moves.shape[0] = {moves.shape[0]}")
        print(f"rewards.shape[0] = {rewards.shape[0]}")
        self.policy_network.fit((boards, players), moves)
        self.value_network.fit((boards, players), rewards)

    #def prepare_state_with_player_info(self, state, moves):
        #for s,m in zip(state, moves):


    def convert_move_to_index(self, move):
        row, col = move
        return row * 8 + col

    def save_training_data(self, training_data):
        serializable_data = [
            (state.tolist(), move, reward) for state, move, reward in training_data
        ]
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as file:
                    existing_data = json.load(file)
            except (json.JSONDecodeError, ValueError):
                existing_data = []
        else:
            existing_data = []

        existing_data.extend(serializable_data)

        with open(self.data_file, 'w') as file:
            json.dump(existing_data, file)

    def load_training_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as file:
                    training_data = json.load(file)
                return [
                    ((np.array(state), player), move, reward) for ((state, player), move, reward) in training_data
                ]
            except (json.JSONDecodeError, ValueError):
                return []
        else:
            return []


# AlphaZero

class AlphaZeroSelfPlay:
    def __init__(self, game, policy_value_network, num_games, n_simulations=2, c_puct=1.4):
        self.game = game
        self.policy_value_network = policy_value_network
        self.num_games = num_games
        self.n_simulations = n_simulations
        self.c_puct = c_puct

    def play_games(self):
        training_data = []
        average_moves = 0
        for _ in range(self.num_games):
            game_data = self.play_single_game()
            training_data.extend(game_data)
            average_moves += len(game_data)
        average_moves /= self.num_games
        return training_data, average_moves

    def play_single_game(self):
        state = self.game.get_initial_state()
        game_data = []
        while not state.is_terminal():
            move, move_probs = self.select_move(state)
            policy_target = np.zeros(64)  # Assuming 64 possible moves
            for move_idx, prob in enumerate(move_probs):
                policy_target[move_idx] = prob
            game_data.append((state.board.copy(), policy_target, 0))  # Value will be updated later
            state = state.sow(move)
        reward = state.get_reward()
        for i in range(len(game_data)):
            game_data[i] = (game_data[i][0], game_data[i][1], reward)
        return game_data

    def select_move(self, state):
        mcts = AlphaZeroMCTS(self.game, self.policy_value_network, self.n_simulations, self.c_puct)
        state_copy = copy.deepcopy(state)
        best_move = mcts.search(state_copy)
        move_probs = np.zeros(64)
        for move, node in mcts.root.children.items():
            move_probs[mcts.convert_move_to_index(move)] = node.visit_count / mcts.root.visit_count
        return best_move, move_probs

    # Not used
    def save_training_data(self, training_data, filename='training_data_alpha.json'):
        try:
            with open(filename, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = []

        existing_data.extend(training_data)
        with open(filename, 'w') as file:
            json.dump(existing_data, file)

    def load_training_data(self, filename='new_training_data.json'):
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as file:
                    training_data = json.load(file)
                return [
                    ((np.array(state), player), move, reward) for ((state, player), move, reward) in training_data
                ]
            except (json.JSONDecodeError, ValueError):
                return []
        else:
            return []

    def train_policy_value_networks(self, training_data):
        #states, policies, values = training_data
        #self.policy_value_network.fit(states, policies, values)
        print(f"Training with {len(training_data)} data points")
        if len(training_data) == 0:
            print("No training data generated.")
            return
        states, policies, values = zip(*training_data)
        boards, players = zip(*states)
        boards = np.array(boards).reshape(-1, 4, 8, 1)  # Adjust shape as needed
        players = np.array(players)
        policies = np.array(policies)
        policies = np.array([self.convert_move_to_index(policy) for policy in policies])
        policies = np.array([np.eye(32)[policy] for policy in policies])
        values = np.array(values)
        print(f"boards.shape[0] = {boards.shape[0]}")
        print(f"players.shape[0] = {players.shape[0]}")
        print(f"policies.shape[0] = {policies.shape[0]}")
        print(f"values.shape[0] = {values.shape[0]}")
        self.policy_value_network.fit([boards, players], policies, values)


    def convert_move_to_index(self, move):
        row, col = move
        return row * 8 + col
