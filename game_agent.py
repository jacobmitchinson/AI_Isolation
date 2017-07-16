"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # get own and opposition move
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # set initial aggression level
    aggression = 2

    # get the number of spaces remaining on the board
    spaces_remaining = float(len(game.get_blank_spaces())) / (game.width * game.height)

    """
        This is a multi tiered approach to applying aggression where aggression
        is highest as the start and lowers as the game goes on.
    """
    if spaces_remaining >= .70:
        aggression = 2.00
    elif spaces_remaining >= .40:
        aggression = 1.50
    elif spaces_remaining >= .15:
        aggression = 1.25

    # return own moves minus opposition moves with aggression applied
    return float(own_moves - aggression * opp_moves)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # get own and opposition moves
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # return score
    return float(own_moves - 2 * opp_moves)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # get the opponent location
    opponent_location = game.get_player_location(game.get_opponent(player))

    # create variables for x and y coordinates
    opponent_x = opponent_location[0]
    opponent_y = opponent_location[1]

    # create the symmetrical move
    symmetrical_move = (game.width - opponent_x - 1, game.height - opponent_y - 1)

    # get own and opposition moves
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    """
        Check if a symmetrical move can be made and increase the score if this
        is the case. Otherwise apply custom_score_2.
    """
    if symmetrical_move in game.get_legal_moves(player):
        return float(own_moves * 10 - (2 * opp_moves))
    else:
        return float(own_moves - (2 * opp_moves))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
            the lectures.

            This should be a modified version of MINIMAX-DECISION in the AIMA text.
            https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

            **********************************************************************
             You MAY add additional methods to this class, or define helper
                  functions to implement the required functionality.
            **********************************************************************

            Parameters
            ----------
            game : isolation.Board
             An instance of the Isolation game `Board` class representing the
             current game state

            depth : int
             Depth is an integer representing the maximum number of plies to
             search in the game tree before aborting

            Returns
            -------
            (int, int)
             The board coordinates of the best move found in the current search;
             (-1, -1) if there are no legal moves

            Notes
            -----
             (1) You MUST use the `self.score()` method for board evaluation
                 to pass the project tests; you cannot call any other evaluation
                 function directly.

             (2) If you use any helper functions (e.g., as shown in the AIMA
                 pseudocode) then you must copy the timer check into the top of
                 each helper function or else your agent will timeout during
                 testing.
        """
        # keep applying iterative deepening until out of time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # take the best position
        return self._minimax(game, depth)[0]

    def _minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # run the function until we timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # return the score if the max depth is reached
        if depth == 0:
            return (None, self.score(game, self))

        # set the maximising player based on the active player
        is_maximising_player = game.active_player == self

        # use a negative or positive values based on maximising condition
        score = float('-inf') if is_maximising_player else float('inf')

        # use the max function if maximising or min if not
        compare = max if is_maximising_player else min

        # default best move
        best_move = (-1, -1)

        # get all legal moves in the game
        moves = game.get_legal_moves()

        """
            Recursively call this function decreasing the depth by one each time.
            This has the effect of looking at all nodes until the timer runs out.
            We get the forecast score from this game simulation and compare it
            with our current high score. We use the compare function which
            will call max or min depending on whether we are on a minimising or
            maximising node.
        """
        for move in moves:
            forecast_score = self._minimax(game.forecast_move(move), depth - 1)[1]

            if compare(score, forecast_score) == forecast_score:
                best_move = move
                score = compare(score, forecast_score)

        return best_move, score

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = (-1, -1)

        # Keep looking deeper until out of time
        depth = 1
        while True:
            try:

                best_move = self.alphabeta(game, depth)
                depth += 1

            except SearchTimeout:
                break

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # apply iterative deepening
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # get the best move from alpha beta pruning
        return self._alphabeta(game, depth)[0]

    """
        This function applies alpha-beta pruning with iteartive deepening
        to find the best move. It is a recursive function that will look
        at each node until it times out.
    """
    def _alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        # return if timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # return score if max depth reached
        if depth == 0:
            return (None, self.score(game, self))

        # set the maximising player based on the active player
        is_maximising_player = game.active_player == self

        # set alpha and beta depending on max or min
        is_alpha = True if is_maximising_player else False
        is_beta =  True if not is_maximising_player else False

        # set negative offset
        score = float('-inf') if is_maximising_player else float('inf')

        # create compare function depending on max or min
        compare = max if is_maximising_player else min

        # default best move
        best_move = (-1, -1)

        # get all legal moves
        moves = game.get_legal_moves()

        """
            Recursively call this function decreasing the depth by one each time.
            This has the effect of looking at all nodes until the timer runs out.
            We get the forecast score from this game simulation and compare it
            with our current high score. We use the compare function which
            will call max or min depending on whether we are on a minimising or
            maximising node. We also keep track of alpha and beta and adjust
            the best score using alpha beta pruning.
        """
        for move in moves:
            forecast_game = game.forecast_move(move)

            forecast_score = self._alphabeta(forecast_game, depth - 1, alpha, beta)[1]

            if compare(score, forecast_score) == forecast_score:
                best_move = move
                score = compare(score, forecast_score)

            if is_alpha:
                if forecast_score >= beta:
                    return (best_move, score)

            if is_beta:
                if forecast_score <= alpha:
                    return (best_move, score)

            if is_alpha:
                alpha = compare(alpha, score)

            if is_beta:
                beta = compare(beta, score)

        return best_move, score
