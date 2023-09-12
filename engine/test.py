# Test utils

import unittest
import os

import numpy as np
from matplotlib import pyplot as plt

from game import MacroMateEngine
from utils import Utils


def get_empty_board(grid_height=32):
    return np.ones((grid_height, grid_height, 2)) * -1


def get_single_piece_board(grid_height=32, piece=0, player=0, position=(0, 0)):
    board = get_empty_board(grid_height)
    board[position[0], position[1], 0] = player
    board[position[0], position[1], 1] = piece
    return board

# Create lines
LINE_DIFFS = np.array([
    # Horizontals
    [(0, i) for i in range(1, 9)],
    [(0, -i) for i in range(1, 9)],
    # Verticals
    [(i, 0) for i in range(1, 9)],
    [(-i, 0) for i in range(1, 9)],
    # Diagonals
    [(i, i) for i in range(1, 9)],
    [(-i, -i) for i in range(1, 9)],
    [(i, -i) for i in range(1, 9)],
    [(-i, i) for i in range(1, 9)],
])
    
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

    def test_precalc_lines(self):
        # Wannabe fuzzing
        for test_i in range(4):
            board = get_empty_board()
            start = np.random.randint(8, 24, (1, 2))
            in_lines = start + LINE_DIFFS.copy()

            # After the first two runs, delete a few lines
            if test_i >= 2:
                # Delete a few lines (but not all)
                num_del = np.random.randint(1, in_lines.shape[0]-1)
                del_lines = np.random.choice(in_lines.shape[0], num_del, replace=False)
                in_lines = np.delete(in_lines, del_lines, axis=0)

            # Check that the precalc lines are correct
            utils = Utils(board)
            clean_lines, start_indices = utils.precalculate_lines(board, in_lines)

            exp_num_pos = in_lines.shape[0] * in_lines.shape[1]
            num_pos = clean_lines.shape[0]
            self.assertEqual(num_pos, exp_num_pos)

            exp_starts = np.array([i for i in range(0, clean_lines.shape[0], in_lines.shape[1])])
            assert np.array_equal(start_indices, exp_starts)

            # Check positions of lines
            for i in range(in_lines.shape[0]):
                for j in range(in_lines.shape[1]):
                    exp_pos = in_lines[i, j]
                    pos = clean_lines[i * in_lines.shape[1] + j]
                    self.assertEqual(pos[0], exp_pos[0])
                    self.assertEqual(pos[1], exp_pos[1])

            # Now block a single line (but not completely)
            blocked_line = np.random.randint(0, in_lines.shape[0] - 1)
            blocked_ind = np.random.randint(1, in_lines.shape[1])
            blocked_pos = in_lines[blocked_line, blocked_ind]
            board[blocked_pos[0], blocked_pos[1], 0] = -2

            # Check that the precalc lines are correct
            utils = Utils(board)
            lines, start_indices = utils.precalculate_lines(board, in_lines)

            start_diff = np.diff(start_indices)
            # Add last diff (which is to the end of the array)
            start_diff = np.append(start_diff, lines.shape[0] - start_indices[-1])

            exp_diff = np.ones_like(start_indices) * in_lines.shape[1]
            exp_diff[blocked_line] = blocked_ind

            assert np.array_equal(start_diff, exp_diff)

            # Check positions of lines: remove blocked parts of line
            cur_ind = 0
            for i in range(in_lines.shape[0]):
                for j in range(in_lines.shape[1]):
                    if i == blocked_line and j >= blocked_ind:
                        continue
                    exp_pos = in_lines[i, j]
                    pos = lines[cur_ind]
                    self.assertEqual(pos[0], exp_pos[0])
                    self.assertEqual(pos[1], exp_pos[1])
                    cur_ind += 1

            # Check complete removal of a line
            blocked_pos = in_lines[blocked_line, 0]
            board[blocked_pos[0], blocked_pos[1], 0] = -2

            # Check that the precalc lines are correct
            utils = Utils(board)
            lines, start_indices = utils.precalculate_lines(board, in_lines)

            start_diff = np.diff(start_indices)

            # Add last diff (which is to the end of the array)
            start_diff = np.append(start_diff, lines.shape[0] - start_indices[-1])

            exp_diff = np.ones(in_lines.shape[0] - 1, dtype=np.int8) * in_lines.shape[1]
            assert np.array_equal(start_diff, exp_diff)

            # Check positions of lines: remove blocked parts of line
            cur_ind = 0
            for i in range(in_lines.shape[0]):
                if i == blocked_line:
                    continue
                for j in range(in_lines.shape[1]):
                    exp_pos = in_lines[i, j]
                    pos = lines[cur_ind]
                    self.assertEqual(pos[0], exp_pos[0])
                    self.assertEqual(pos[1], exp_pos[1])
                    cur_ind += 1

    def test_precalc_lines_outside(self):
        # Just follow the edge and check if correct lines are returned
        board = get_empty_board()
        utils = Utils(board)
        
        # Check four corners
        corners = np.array([
            (0, 0),
            (0, board.shape[0] - 1),
            (board.shape[0] - 1, 0),
            (board.shape[0] - 1, board.shape[0] - 1),
        ])
        for corner in corners:
            lines, start_indices = utils.precalculate_lines(board, corner + LINE_DIFFS.copy())
            self.assertEqual(lines.shape[0], 3 * 8)

            # Check start indices
            exp_start = np.array([0, 8, 16])
            np.testing.assert_array_equal(start_indices, exp_start)

            # Remove coords out of bounds
            exp_lines = corner + LINE_DIFFS.copy()
            exp_lines = exp_lines.reshape((-1, 2))
            exp_lines = exp_lines[exp_lines[:, 0] >= 0]
            exp_lines = exp_lines[exp_lines[:, 1] >= 0]
            exp_lines = exp_lines[exp_lines[:, 0] < board.shape[0]]
            exp_lines = exp_lines[exp_lines[:, 1] < board.shape[0]]
            np.testing.assert_array_equal(lines, exp_lines)

        # Pick some random positions close to the edge
        bs = board.shape[0]
        options = [0, 1, 2, 3, 4, bs - 1, bs - 2, bs - 3, bs - 4]
        for __ in range(4):
            pos = np.random.choice(options, 2)
            lines, start_indices = utils.precalculate_lines(board, pos + LINE_DIFFS.copy())

            # remove all coords out of bounds
            exp_lines = pos + LINE_DIFFS.copy()
            exp_lines = exp_lines.reshape((-1, 2))
            exp_lines = exp_lines[exp_lines[:, 0] >= 0]
            exp_lines = exp_lines[exp_lines[:, 1] >= 0]
            exp_lines = exp_lines[exp_lines[:, 0] < board.shape[0]]
            exp_lines = exp_lines[exp_lines[:, 1] < board.shape[0]]
            np.testing.assert_array_equal(lines, exp_lines)

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
            pre_lines, start_indices = utils.all_moves[pos[0]][pos[1]][piece]

            # Place a single queen of the player
            board[pos[0], pos[1], 0] = player
            board[pos[0], pos[1], 1] = piece

            # Check unblocked
            own_mask, other_mask = utils.get_masks(board, player)
            unblocked_lines = utils.check_lines(pre_lines, start_indices, own_mask, other_mask)
            num_unblocked = unblocked_lines.shape[0]

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

            # Check that the line is blocked by the other player
            own_mask, other_mask = utils.get_masks(board, player)
            blocked_lines = utils.check_lines(pre_lines, start_indices, own_mask, other_mask)
            num_blocked = blocked_lines.shape[0]
            self.assertEqual(num_blocked, 8 * dist, msg=f"OTHER WRONG pos: {pos} dist: {dist}")
            self.assertTrue(num_unblocked > num_blocked)

            # # Check that the line is blocked by the own player
            # # Convert pawns to own player
            board[ppos[:, 0], ppos[:, 1], 0] = 0
            own_mask, other_mask = utils.get_masks(board, player)
            blocked_lines = utils.check_lines(pre_lines, start_indices, own_mask, other_mask)
            self.assertEqual(blocked_lines.shape[0], 8 * (dist - 1), msg=f"OWN WRONG pos: {pos} dist: {dist}")

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
            pre_lines, start_indices = utils.all_moves[pos[0]][pos[1]][4]
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
                blocked_lines = utils.check_lines(pre_lines, start_indices, own_mask, other_mask)
                self.assertEqual(blocked_lines.shape[0], 3 * dist, msg=f"OTHER WRONG pos: {pos} dist: {dist}")

                # Convert pawns to own player
                board[ppos[:, 0], ppos[:, 1], 0] = 0
                own_mask, other_mask = utils.get_masks(board, 0)
                blocked_lines = utils.check_lines(pre_lines, start_indices, own_mask, other_mask)
                self.assertEqual(blocked_lines.shape[0], 3 * (dist - 1), msg=f"OWN WRONG pos: {pos} dist: {dist}")


if __name__ == '__main__':
    unittest.main()