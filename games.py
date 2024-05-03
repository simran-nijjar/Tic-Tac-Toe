# games.py
# Simran Nijjar

import copy
import itertools
from collections import namedtuple

import numpy as np

# Defines the different algorithms used to choose the computer's moves 
# The algorithms are MinMax, Expectimax, and AlphaBeta Search

GameState = namedtuple('GameState', 'to_move, utility, board, moves')

def gen_state(to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    # Given whose turn it is to move, the positions of X's and O's on the board, and number of rows, columns return the game state"""

    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, utility=0, board=board, moves=moves)

# MinMax Search
def minmax(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))


def minmax_cutoff(game, state, d = 4, cutoff_test = None, eval_fn = None):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the cutoff depth. At that level use evaluation func."""
    player = game.to_move(state)

    def max_value(state, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), depth - 1))
        return v

    def min_value(state, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), depth - 1))
        return v
    
    # Return true if the current depth is greater than the given depth or if the game is in a terminal state
    def calc_cutoff_test(state, depth):
        return d < depth or game.terminal_test(state)
    
    # Returns the evaluation func
    def calc_eval_fn(state):
        return game.evaluation_func(state, player)

    # Body of minmax_decision:
    cutoff_test = calc_cutoff_test
    eval_fn = calc_eval_fn
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), d))

# Expectimax
def expectimax(game, state):
    """
    [Figure 5.11]
    Return the best move for a player after dice are thrown. The game tree
	includes chance nodes along with min and max nodes.
	"""
    player = game.to_move(state)

    def max_value(state):
        v = -np.inf
        for a in game.actions(state):
            v = max(v, chance_node(state, a))
        return v

    def min_value(state):
        v = np.inf
        for a in game.actions(state):
            v = min(v, chance_node(state, a))
        return v

    def chance_node(state, action):
        res_state = game.result(state, action)
        if game.terminal_test(res_state):
            return game.utility(res_state, player)
        sum_chances = 0
        num_chances = len(game.chances(res_state))
        for chance in game.chances(res_state): 
            chanceCalc = 0
            if res_state.to_move != player: #If the move is the oppenents
                chanceCalc = min_value(res_state) #Calculate for minimization
            else: #If the move is the players
                chanceCalc = max_value(res_state) #Calculate for maximization
            sum_chances += chanceCalc * (1 * chance) #Here the probability I have randomly assigned 1 for each node
        if (num_chances == 0): #If the number of chances is zero return 0
            return 0
        return sum_chances / num_chances; #Else return the number of chances

    # Body of expectimax:
    return max(game.actions(state), key=lambda a: chance_node(state, a), default=None)

def expectimax_cutoff(game, state, d = 4, cutoff_test = None, eval_fn = None):
    player = game.to_move(state)

    def max_value(state, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, chance_node(state, a))
        return v

    def min_value(state, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, chance_node(state, a))
        return v

    def chance_node(state, action):
        res_state = game.result(state, action)
        if game.terminal_test(res_state):
            return game.utility(res_state, player)
        sum_chances = 0
        num_chances = len(game.chances(res_state))
        for chance in game.chances(res_state): 
            chanceCalc = 0
            if res_state.to_move != player: #If the move is the oppenents
                chanceCalc = min_value(res_state ,d) #Calculate for minimization
            else: #If the move is the players
                chanceCalc = max_value(res_state, d) #Calculate for maximization
            sum_chances += chanceCalc * (1 * chance) #Here the probability I have randomly assigned 1 for each node
        if (num_chances == 0): #If the number of chances is zero return 0
            return 0
        return sum_chances / num_chances; #Else return the number of chances

    # Return true if the current depth is greater than the given depth or if the game is in a terminal state
    def calc_cutoff_test(state, depth):
        return d < depth or game.terminal_test(state)
    
    # Returns the evaluation func
    def calc_eval_fn(state):
        return game.evaluation_func(state, player)

    # Body of expectimax:
    cutoff_test = calc_cutoff_test
    eval_fn = calc_eval_fn
    return max(game.actions(state), key=lambda a: chance_node(state, a), default=None)

# Alpha Beta
def alpha_beta_search(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for action in game.actions(state): #Finding action in game state
            v = max(v, min_value(game.result(state, action), alpha, beta))
            if (v >= beta):
                return v
            alpha = max(alpha, v) #max's best option on path to root
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for action in game.actions(state):  #Finding action in game state
            v = min(v, max_value(game.result(state, action), alpha, beta))
            if (v <= alpha):
                return v
            beta = min(beta, v) #min's best option on path to root
        return v

    # Body of alpha_beta_search:
    best_action = None
    alpha = -np.inf #Initialized to negative infinity to find max value
    beta = np.inf #Initialized to positive infinity to find min value
    for action in game.actions(state): #Finding action in game state
        v = min_value(game.result(state, action), alpha, beta) #Use min_value() function to check for the opponent's moves
        if (v > alpha):
            best_action = action
            alpha = v
    return best_action


def alpha_beta_cutoff_search(game, state, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for action in game.actions(state): #Finding action in game state
            v = max(v, min_value(game.result(state, action), alpha, beta, depth + 1))
            if (v >= beta):
                return v
            alpha = max(alpha, v) #max's best option on path to root
        return v

    def min_value(state, alpha, beta, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for action in game.actions(state):  #Finding action in game state
            v = min(v, max_value(game.result(state, action), alpha, beta, depth + 1))
            if (v <= alpha):
                return v
            beta = min(beta, v) #min's best option on path to root
        return v
    
    # Return true if the current depth is greater than the given depth or if the game is in a terminal state
    def calc_cutoff_test(state, depth):
        return d < depth or game.terminal_test(state)
    
    # Returns the evaluation func
    def calc_eval_fn(state):
        return game.evaluation_func(state, player)

    # Body of alpha_beta_cutoff_search starts here:
    best_action = None
    alpha = -np.inf
    beta = np.inf
    cutoff_test = calc_cutoff_test
    eval_fn = calc_eval_fn
    for action in game.actions(state):
        v = min_value(game.result(state, action), alpha, beta, d)
        if v > alpha:
            alpha = v
            best_action = action
    return best_action

# Players for Games
def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move

def alpha_beta_player(game, state):
    if( game.d == -1):
        return alpha_beta_search(game, state)
    return alpha_beta_cutoff_search(game, state)


def minmax_player(game,state):
    if( game.d == -1):
        return minmax(game, state)
    return minmax_cutoff(game, state)


def expectimax_player(game, state):
    if( game.d == -1):
        return expectimax(game, state)
    return expectimax_cutoff(game, state)


# Game class
class Game:
    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))


# TicTacToe class
class TicTacToe(Game):
    # Player is 'X' or 'O'.
    # depth = -1 means max search tree depth to be used."""

    def __init__(self, h=3, v=3, k=3, d=-1):
        self.h = h
        self.v = v
        self.k = k
        self.depth = d
        moves = [(x, y) for x in range(1, h + 1)
                 for y in range(1, v + 1)]
        self.initial = GameState(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            return self.k if player == 'X' else -self.k
        else:
            return 0

    def evaluation_func(self, state, player):
        """computes value for a player on board after move.
            Likely it is better to conside the board's state from 
            the point of view of both 'X' and 'O' players and then subtract
            the corresponding values before returning.""" 
        # Determine who is the player and who is the opponent
        if player == 'X':
            opponent = 'O'
        else:
            opponent = 'X'
        # Compute player score
        playerScore = 0
        opponentScore = 0
        playerScore = self.evaluation_calc(state.board, player)
        opponentScore = self.evaluation_calc(state.board, opponent)
        # Compute player points, points are calculated based on how large the score is
        playerPoints = self.calculate_points(playerScore)
        opponentPoints = self.calculate_points(opponentScore)
        return abs(playerPoints - opponentPoints)

    def evaluation_calc(self, board, player):
        "Helper evaluation calulation method for evaluation_func to count the number of moves for the player"
        score = 0
        rows = self.h
        cols = self.v
        #Count how many moves are in the board already
        for row in range(1, rows + 1):          
            if self.k_in_row(board, (row, 1), player, (1, 0)):
                score += 1
        for col in range(1, cols + 1):
            #Horizontal
            if self.k_in_row(board, (1, col), player, (0, 1)):
                score += 1     
        #Diagonal to the left
        if self.k_in_row(board, (1,1), player, (1, 1)):
            score += 1                
        #Diagonal to the right
        if self.k_in_row(board, (1, cols), player, (1, -1)):
            score += 1
        return score
    
    def calculate_points(self, count):
        "Helper point calculation method for evaluation_func to calculate points for each player"
        #Return 0 if the player score is 0
        if (count == 0):
            return 0
        #Multiply points by 10 for every score the player has
        points = 1
        for i in range(1, count):
            points *= (i * 10)
        return points
		
    def k_in_row(self, board, move, player, delta_x_y):
        """Return true if there is a line through move on board for player.
        hint: This function can be extended to test of n number of items on a line 
        not just self.k items as it is now. """
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= self.k

    def chances(self, state):
        """Return a list of all possible states."""
        chances = []
        return chances