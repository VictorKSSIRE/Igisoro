import numpy as np
import sys
import copy


class Igisoro:
    def __init__(self):
        # Initialize the board with the correct initial display using NumPy arrays:
        # Player 1's rows: back row (0s) and front row (4s)
        # Player 0's rows: front row (4s) and back row (0s)
        self.board = np.zeros((4, 8), dtype=int)
        self.board[1, :] = 4
        self.board[2, :] = 4
        self.current_player = 0  # Player 0 starts

    def get_curr_state(self):
        return self

    def get_initial_state(self):
        self.board = np.zeros((4, 8), dtype=int)
        self.board[1, :] = 4
        self.board[2, :] = 4
        self.current_player = 0  # Player 0 starts
        return self

    def display_board(self):
        print("Current board state:")
        # Display the board
        for row in self.board:
            print(' '.join(f"{pit:2d}" for pit in row))
        print(f"Current Player => Player {self.current_player}")
        print()

    def valid_move(self, row_index, col_index):
        # Check if the move is valid: correct side of the board, and pit has more than one bead
        if self.current_player == 0 and row_index in [2, 3] and self.board[row_index, col_index] > 1:
            return True
        if self.current_player == 1 and row_index in [0, 1] and self.board[row_index, col_index] > 1:
            return True
        return False

    def get_legal_moves(self):
        moves = []
        for row in range(4):
            for col in range(8):
                if self.valid_move(row, col):
                    moves.append((row, col))
        return moves

    def reap(self, row, col):
        player_row_1 = (self.current_player == 0) and (row == 2)
        player_row_2 = (self.current_player == 1) and (row == 1)
        opponent_rows = [0, 1] if self.current_player == 0 else [2, 3]
        if (player_row_1 or player_row_2) and self.board[row, col] > 1:
            if all(self.board[opp_row, col] > 0 for opp_row in opponent_rows):
                # Take all beads from the opponent's columns and continue sowing
                beads_to_reap = sum(self.board[opp_row, col] for opp_row in opponent_rows)
                for opp_row in opponent_rows:
                    self.board[opp_row, col] = 0
                return beads_to_reap
        return 0

    def sow(self, position):
        nbr_sow = 0
        row_index, col_index = position
        if not self.valid_move(row_index, col_index):
            #print("Invalid move, try again.")
            return copy.deepcopy(self)

        beads_to_distribute = self.board[row_index, col_index]
        self.board[row_index, col_index] = 0
        row, col = copy.deepcopy((row_index, col_index))

        reap_position = (row, col)
        while beads_to_distribute > 0:
            # Determine the next position
            if (row == 2 or row == 0) and col != 0:
                col -= 1
            elif (row == 2 or row == 0) and col == 0:
                row += 1
            elif (row == 3 or row == 1) and col != 7:
                col += 1
            elif (row == 3 or row == 1) and col == 7:
                row -= 1

            # Sow the bead
            self.board[row, col] += 1
            beads_to_distribute -= 1

            # If the last bead lands in a non-empty pit, reap or pick up those beads and continue sowing
            if beads_to_distribute == 0 and self.board[row, col] > 1:
                beads_to_distribute = self.reap(row, col)
                if beads_to_distribute > 0:
                    row, col = reap_position
                else:
                    nbr_sow += 1
                    beads_to_distribute = self.board[row, col]
                    reap_position = (row, col)
                self.board[row, col] = 0
            if nbr_sow > 1000:
                "Sowing Took more than 1000 steps, therefore assume win"
                break
        self.current_player = 1 - self.current_player

        return copy.deepcopy(self)

    def has_valid_moves(self, player):
        rows = [2, 3] if player == 0 else [0, 1]
        for row in rows:
            for col in range(8):
                if self.valid_move(row, col):
                    return True
        return False

    def is_terminal(self):
        # The game is terminal if either player has no valid moves left
        return not self.has_valid_moves(self.current_player)

    def check_win(self):
       # Game ends if the current player has no valid moves left
        if not self.has_valid_moves(self.current_player):
            return True
        return False

    def get_reward(self):
        # If player 0 wins, positive reward
        if self.current_player == 1 and not self.has_valid_moves(self.current_player):
            return 1
        # If player 1 wins, negative reward
        elif self.current_player == 0 and not self.has_valid_moves(self.current_player):
            return -1
        # If neither player has won, i.e. the game is still ongoing, no reward
        return 0

    def input_moves(self, positions):
        """
        Function to input a set of moves for Player 0 and Player 1 sequentially.
        """
        for position in positions:
            if self.check_win():
                break
            if not self.sow(position):
                print(f"Invalid move for Player {self.current_player} at position {position}.")
                break
            self.current_player = 1 - self.current_player
            self.display_board()

    def play_game(self):
        while not self.check_win():
            self.display_board()
            position_input = input(
                f"Player {self.current_player}, choose a position (row, column) as a tuple or Shift+Q to quit: ")

            if position_input.upper() == 'Q':
                print("Game terminated by user.")
                sys.exit()

            try:
                position = tuple(map(int, position_input.strip("()").split(",")))
            except ValueError:
                print("Invalid input format. Please enter a tuple in the format (row, column).")
                continue

            self.sow(position)

        print("Game over!")
        self.display_board()


def main():
    game = Igisoro()
    #Set of moves to prepare game
    #moves = [(2, 2), (1, 7), (2, 1), (1, 0), (2, 7), (1, 2), (2, 0), (1, 1), (2, 5), (1, 5)]
    #game.input_moves(moves)
    game.play_game()


if __name__ == "__main__":
    main()
