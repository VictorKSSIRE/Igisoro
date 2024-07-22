import os
import copy
import numpy as np
import random
import time
from selfplay import AlphaZeroSelfPlay
from igisoro import Igisoro
from mcts import AlphaZeroMCTS
from network import PolicyValueNetwork


if __name__ == "__main__":
    game = Igisoro()
    policy_value_network = PolicyValueNetwork()
    self_play = AlphaZeroSelfPlay(game, policy_value_network, num_games=100)  # Should be 100
    total_training_time = 0
    #breakpoint()
    all_training_data = self_play.load_training_data()  # Load all saved training data

    for epoch in range(1):  # Number of training epochs
        print(f"Start")
        start_time = time.time()  # Start timer

        self_play.train_policy_value_networks(all_training_data)

        end_time = time.time()  # End timer
        epoch_duration = end_time - start_time
        total_training_time += epoch_duration

        print(f"End => Duration: {epoch_duration:.2f} seconds")
        # Save the model
        if not os.path.exists('models'):
            os.makedirs('models')
        policy_model_path = f'models/policy_value_network_epoch_updated.keras'
        policy_value_network.model.save(policy_model_path)

    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"Average Epoch Duration: {total_training_time / 10:.2f} seconds")
