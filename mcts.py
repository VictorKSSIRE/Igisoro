import numpy as np
import random

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, c_param=1.4):
        choices_weights = {
            move: (child.value_sum / child.visit_count) +
            c_param * np.sqrt((2 * np.log(self.visit_count) / child.visit_count)) for move, child in self.children.items()
        }
        best_move = max(choices_weights, key=choices_weights.get)
        return self.children[best_move]

    def most_visited_child_move(self):
        most_visited_move = max(self.children, key=lambda move: self.children[move].visit_count)
        return most_visited_move


class MCTS:
    def __init__(self, game, n_simulations):
        self.game = game
        self.n_simulations = n_simulations

    def search(self, initial_state):
        #breakpoint()
        root = MCTSNode(initial_state)
        for _ in range(self.n_simulations):
            node = self.select(root)
            #breakpoint()
            reward = self.simulate(node)
            #breakpoint()
            self.backpropagate(node, reward)
        #breakpoint()
        return root.most_visited_child_move()

    def select(self, node):
        #breakpoint()
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, node):
        #breakpoint()
        move = random.choice(node.state.get_legal_moves())
        next_state = node.state.sow(move)
        child_node = MCTSNode(next_state, node)
        node.children[move] = child_node
        return child_node

    def simulate(self, node):
        #breakpoint()
        current_state = node.state
        while not current_state.is_terminal():
            move = random.choice(current_state.get_legal_moves())
            current_state = current_state.sow(move)
        #breakpoint()
        return current_state.get_reward()

    def backpropagate(self, node, reward):
        #breakpoint()
        while node is not None:
            node.visit_count += 1
            node.value_sum += reward
            node = node.parent

# AlphaZero


class AlphaZeroMCTSNode:
    def __init__(self, state, parent=None, prior=0, depth=0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.depth = depth

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, c_param=1.4, epsilon=1e-6):
        choices_weights = {
            move: (child.value_sum / (child.visit_count + epsilon)) +
                  c_param * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            for move, child in self.children.items()
        }
        best_move = max(choices_weights, key=choices_weights.get)
        return self.children[best_move]

    def most_visited_child_move(self):
        most_visited_move = max(self.children, key=lambda move: self.children[move].visit_count)
        return most_visited_move


class AlphaZeroMCTS:
    def __init__(self, game, policy_value_network, n_simulations, c_puct=1.4, max_depth=50):
        self.game = game
        self.policy_value_network = policy_value_network
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.root = None

    def search(self, initial_state):
        self.root = AlphaZeroMCTSNode(initial_state)
        for _ in range(self.n_simulations):
            node = self.select(self.root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        return self.root.most_visited_child_move()

    def select(self, node):
        while not node.state.is_terminal() and node.depth < self.max_depth:
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child(c_param=self.c_puct)
        return node

    def expand(self, node):
        legal_moves = node.state.get_legal_moves()
        state_array = (np.array(node.state.board).reshape(1, 4, 8), np.array([node.state.current_player]))
        move_probs, _ = self.policy_value_network.predict(state_array)
        move_probs = move_probs.flatten()
        for move in legal_moves:
            if move not in node.children:
                move_prob = move_probs[self.convert_move_to_index(move)]
                next_state = node.state.sow(move)
                node.children[move] = AlphaZeroMCTSNode(next_state, node, prior=move_prob, depth=node.depth + 1)
        return random.choice(list(node.children.values()))

    def simulate(self, node):
        current_state = node.state
        steps = 0
        max_steps = 500  # Safeguard against infinite loops
        while not current_state.is_terminal() and steps < max_steps:
            state_array = np.array(current_state.board).reshape(1, 4, 8)
            move_probs, _ = self.policy_value_network.predict(state_array)
            move_probs = move_probs.flatten()
            legal_moves = current_state.get_legal_moves()
            move_probs = [move_probs[self.convert_move_to_index(move)] for move in legal_moves]
            move = random.choices(legal_moves, weights=move_probs, k=1)[0]
            current_state = current_state.sow(move)
            steps += 1

        if steps == max_steps:
            print("Simulation stopped early due to step limit.")

        state_array = np.array(current_state.board).reshape(1, 4, 8)
        _, value = self.policy_value_network.predict(state_array)
        value = value[0][0]
        return value

    def backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.value_sum += reward
            node = node.parent

    def convert_move_to_index(self, move):
        row, col = move
        return row * 8 + col  # Assuming an 8x8 board
