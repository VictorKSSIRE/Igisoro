import os
import copy
import numpy as np
import random
import time
from selfplay import SelfPlay
from igisoro import Igisoro
from mcts import MCTS, MCTSNode
from network import PolicyNetwork, ValueNetwork


if __name__ == "__main__":
    game = Igisoro()
    policy_network = PolicyNetwork()
    value_network = ValueNetwork()
    self_play = SelfPlay(game, policy_network, value_network, n_games=100)  # Should be 100
    overall_average_moves = 0
    total_training_time = 0

    for epoch in range(1):  # Number of training epochs
        print(f"Start")
        start_time = time.time()  # Start timer
        #training_data, average_moves = self_play.play_games()
        #self_play.save_training_data(training_data)  # Save the training data
        all_training_data = self_play.load_training_data()  # Load all saved training data
        self_play.train_policy_value_networks(all_training_data)

        end_time = time.time()  # End timer
        epoch_duration = end_time - start_time
        total_training_time += epoch_duration

        print(f"End => Duration: {epoch_duration:.2f} seconds")
        #overall_average_moves += average_moves
        # Save the models
        if not os.path.exists('models'):
            os.makedirs('models')
        policy_model_path = f'models/policy_network_epoch_updated.keras'
        value_model_path = f'models/value_network_epoch_updated.keras'
        policy_network.model.save(policy_model_path)
        value_network.model.save(value_model_path)

    #overall_average_moves /= 10
    #print(f"Average Moves Per Game: {overall_average_moves}")
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"Average Epoch Duration: {total_training_time / 10:.2f} seconds")
