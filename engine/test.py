import unittest

import numpy as np

from game import MacroMateEngine
from utils import Utils


def get_empty_board(grid_height=32):
    return np.ones((grid_height, grid_height, 2)) * -1

    
class TestEngine(unittest.TestCase):
    def test_engine_init(self):
        engine = MacroMateEngine(4)

        self.assertEqual(engine.num_players, 4)
        land_grid = engine.land_grid
        grid_height = land_grid.shape[0]
        self.assertEqual(engine.grid_height, grid_height)
        self.assertEqual(engine.inland_grid.shape, (grid_height, grid_height))
        self.assertEqual(type(engine.utils), Utils)

        # Check board state
        self.assertEqual(engine.board_state.shape, (grid_height, grid_height, 2))

        # All territory <= -1 pieces are -1
        self.assertTrue((engine.board_state[:, :, 0] <= -1).all())
        self.assertTrue((engine.board_state[:, :, 1] == -1).all())
    
    def test_engine_setup_random_starts(self):
        engine = MacroMateEngine(8)
        engine.setup_random_starts()

        board_state = engine.board_state

        # Check that all players have 9 territories
        for player in range(engine.num_players):
            self.assertEqual(np.sum(board_state[:, :, 0] == player), 9)

        # Check pieces: 9 pieces per player
        self.assertEqual(np.sum(board_state[:, :, 1] == 0), 4 * engine.num_players)
        self.assertEqual(np.sum(board_state[:, :, 1] == 1), engine.num_players)
        self.assertEqual(np.sum(board_state[:, :, 1] == 2), engine.num_players)
        self.assertEqual(np.sum(board_state[:, :, 1] == 3), engine.num_players)
        self.assertEqual(np.sum(board_state[:, :, 1] == 4), engine.num_players)
        self.assertEqual(np.sum(board_state[:, :, 1] == 5), engine.num_players)


class TestUtils(unittest.TestCase):

    def test_setup(self):
        board = get_empty_board()
        utils = Utils(board)
        self.assertEqual(utils.grid_height, board.shape[0])

        # Check precalc sizes
        self.assertEqual(len(utils.all_moves), utils.grid_height)
        self.assertEqual(len(utils.all_moves[0]), utils.grid_height)
        self.assertEqual(len(utils.all_moves[0][0]), 6)

    def test_precalc_moves(self):
        board = get_empty_board()
        # Make 1/4 of the board water
        board[16:, 16:, 0] = -2
        utils = Utils(board)

        # Create a 2D array of valid moves (within the board and not water)
        valid_moves = np.random.randint(0, 16, (16, 2))
        invalid = np.random.randint(16, 32, (17, 2))
        invalid2 = np.random.randint(32, 64, (18, 2))
        invalid3 = np.random.randint(-64, 0, (19, 2))
        moves = np.concatenate((valid_moves, invalid, invalid2, invalid3), axis=0)

        # Check that the precalc moves are correct
        pre = utils.precalculate_moves(board, moves)
        assert np.array_equal(pre, valid_moves)

        # Shuffle all moves along the first axis and check if still valid
        np.random.shuffle(moves)
        pre = utils.precalculate_moves(board, moves)
        valid_moves = np.sort(valid_moves, axis=0)
        pre = np.sort(pre, axis=0)
        assert np.array_equal(pre, valid_moves)

    def test_check_lines(self):
        # Other player should block the line on the same field
        # Own player should block the line on the field before

        board = get_empty_board()
        utils = Utils(board)

        for __ in range(12):
            board = get_empty_board()
            pos = np.random.randint(8, board.shape[0] - 8, 2)
            player = 0
            piece = 4  # Queen

            # Place a single queen of the player
            board[pos[0], pos[1], 0] = player
            board[pos[0], pos[1], 1] = piece

            # Check unblocked
            own_mask, other_mask = utils.get_masks(board, player)
            unblocked_moves = utils.get_moves_position(board, pos, own_mask, other_mask)
            num_unblocked = unblocked_moves.shape[0]

            self.assertEqual(num_unblocked, 64)

            # Surround with enemy pawns
            dist = np.random.randint(1, 8)
            ppos = np.array([
                (pos[0] - dist, pos[1] - dist),
                (pos[0] - dist, pos[1]),
                (pos[0] - dist, pos[1] + dist),
                (pos[0], pos[1] - dist),
                (pos[0], pos[1] + dist),
                (pos[0] + dist, pos[1] - dist),
                (pos[0] + dist, pos[1]),
                (pos[0] + dist, pos[1] + dist),
            ])
            board[ppos[:, 0], ppos[:, 1], 0] = 1
            board[ppos[:, 0], ppos[:, 1], 1] = 0

            # Check that the line is blocked by the other player = 8 * dist
            own_mask, other_mask = utils.get_masks(board, player)
            blocked_moves = utils.get_moves_position(board, pos, own_mask, other_mask)
            num_blocked = blocked_moves.shape[0]
            self.assertEqual(num_blocked, 8 * dist, msg=f"OTHER WRONG pos: {pos} dist: {dist}")
            
            # Check that the line is blocked by the own player
            # Convert pawns to own player
            board[ppos[:, 0], ppos[:, 1], 0] = 0
            own_mask, other_mask = utils.get_masks(board, player)
            blocked_moves = utils.get_moves_position(board, pos, own_mask, other_mask)
            num_blocked = blocked_moves.shape[0]
            self.assertEqual(num_blocked, 8 * (dist - 1), msg=f"OWN WRONG pos: {pos} dist: {dist}")

    def test_check_lines_corner(self):
        board = get_empty_board()
        utils = Utils(board)
        corners = [
            ((0, 0), (1, 1)),
            ((0, board.shape[0] - 1), (1, -1)),
            ((board.shape[0] - 1, 0), (-1, 1)),
            ((board.shape[0] - 1, board.shape[0] - 1), (-1, -1)),
        ]

        for pos, direction in corners:
            for dist in range(7, 0, -1):
                board = get_empty_board()
                board[pos[0], pos[1], 0] = 0
                board[pos[0], pos[1], 1] = 4

                # Surround with enemy pawns
                ppos = np.array([
                    (pos[0], pos[1] + dist * direction[1]),
                    (pos[0] + dist * direction[0], pos[1]),
                    (pos[0] + dist * direction[0], pos[1] + dist * direction[1]),
                ])
                board[ppos[:, 0], ppos[:, 1], 0] = 1
                board[ppos[:, 0], ppos[:, 1], 1] = 0

                own_mask, other_mask = utils.get_masks(board, 0)
                moves = utils.get_moves_position(board, pos, own_mask, other_mask)
                self.assertEqual(moves.shape[0], 3 * dist, msg=f"OTHER WRONG pos: {pos} dist: {dist}")

                # Convert pawns to own player
                board[ppos[:, 0], ppos[:, 1], 0] = 0
                own_mask, other_mask = utils.get_masks(board, 0)
                moves = utils.get_moves_position(board, pos, own_mask, other_mask)
                self.assertEqual(moves.shape[0], 3 * (dist - 1), msg=f"OWN WRONG pos: {pos} dist: {dist}")


if __name__ == '__main__':
    unittest.main()