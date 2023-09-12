import numpy as np
from matplotlib import pyplot as plt

from multiprocessing import Pool

DEPTH_SEARCH = 3

# PIECE MOVEMENT (max 8 steps)
PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
MAX_STEPS = 8
diagonal_neighbours = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.int8)
straight_neighbours = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]], dtype=np.int8)
all_neighbours = np.vstack((diagonal_neighbours, straight_neighbours))

# Starting right, clockwise
straight_lines = np.array([
    [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0]], # Right
    [[0, -1], [0, -2], [0, -3], [0, -4], [0, -5], [0, -6], [0, -7], [0, -8]], # Down
    [[-1, 0], [-2, 0], [-3, 0], [-4, 0], [-5, 0], [-6, 0], [-7, 0], [-8, 0]], # Left
    [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8]], # Up
], dtype=np.int8)

diagonal_lines = np.array([
    [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7 ,7], [8, 8]], # Right up
    [[-1, 1], [-2, 2], [-3, 3], [-4, 4], [-5, 5], [-6, 6], [-7 ,7], [-8, 8]], # Right down
    [[-1, -1], [-2, -2], [-3, -3], [-4, -4], [-5, -5], [-6, -6], [-7 ,-7], [-8, -8]], # Left down
    [[1, -1], [2, -2], [3, -3], [4, -4], [5, -5], [6, -6], [7 ,-7], [8, -8]], # Left up
], dtype=np.int8)
queen_lines = np.vstack((straight_lines, diagonal_lines))

pawn_moves = straight_neighbours
pawn_captures = diagonal_neighbours
knight_moves = np.array([[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]], dtype=np.int8)
king_moves = all_neighbours

piece_values = np.array([1, 2, 5, 6, 10, 100], dtype=np.int8)


class Utils:
    def __init__(self, board_state):
        self.initial_board_state = board_state.copy()
        self.grid_height = board_state.shape[0]

        # Precalculate all possible moves
        self.all_moves = []
        for x in range(self.initial_board_state.shape[0]):
            x_moves = []
            for y in range(self.initial_board_state.shape[1]):
                position = np.array([x, y], dtype=np.int8)
                x_moves.append([self.precalculate_piece(position, piece) for piece in range(6)])
            self.all_moves.append(x_moves)

    def precalculate_piece(self, position, piece):
        if piece == 0:
            pos = self.precalculate_moves(self.initial_board_state, position + pawn_moves)
            pos2 = self.precalculate_moves(self.initial_board_state, position + pawn_captures)
            return [pos, pos2]
        elif piece == 1:
            return self.precalculate_moves(self.initial_board_state, position + knight_moves)
        elif piece == 2:
            return self.precalculate_lines(self.initial_board_state, diagonal_lines.copy() + position)
        elif piece == 3:
            return self.precalculate_lines(self.initial_board_state, straight_lines.copy() + position)
        elif piece == 4:
            return self.precalculate_lines(self.initial_board_state, queen_lines.copy() + position)
        elif piece == 5:
            return self.precalculate_moves(self.initial_board_state, position + king_moves)
        else:
            raise Exception("Invalid piece type", piece)

    def search_tree(self, board_state, player, distance=8, depth=DEPTH_SEARCH, alpha=-np.inf, beta=np.inf, is_player=True):
        is_root = depth == DEPTH_SEARCH
        
        # Get the best move for a player using the minimax algorithm with alpha-beta pruning
        if depth == 0:
            # If we've reached the maximum depth, evaluate the board state and return the score
            return self.evaluate_board(board_state, player)

        # Generate all possible moves for the current player
        moves = self.get_moves_player(board_state, player) if is_player else self.get_moves_enemies(board_state, player, distance)

        # Shuffle player moves
        if is_player:
            np.random.shuffle(moves)

        
        # If there are no possible moves, return the worst score
        if moves is None:
            return -np.inf if is_player else np.inf

        # Initialize the best score and move
        best_score = -np.inf if is_player else np.inf
        best_move = None

        # Loop over all possible moves and recursively search the game tree
        for move in moves:
            # Execute the move on a copy of the board state
            board_state_copy = self.execute_move(board_state, move, inplace=False)

            # Recursively search the game tree for the other player
            score = self.search_tree(board_state_copy, player, distance, depth=depth-1, alpha=alpha, beta=beta, is_player=not is_player)

            # Update the best score and move based on the current player
            if is_player:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha:
                    break

        # Return the best score and move for the current player
        if is_root:
            return best_move
        else:
            return best_score

    def get_moves_player(self, board_state, player):
        own_mask, other_mask = self.get_masks(board_state, player)
        moves = []
        for piece in np.argwhere(own_mask):
            m = self.get_moves_position(board_state, piece, own_mask, other_mask)
            if m is not None:
                # Add position to move
                m = np.hstack((np.tile(piece, (m.shape[0], 1)), m))
                moves.append(m)
        
        if moves:
            return np.vstack(moves)
        else:
            return np.array([])

    def get_moves_enemies(self, board_state, player, distance=-1):
        enemy_pieces, __ = self.get_enemy_pieces(board_state, player, distance)

        if enemy_pieces is None:
            return None

        # Also calc invalid moves (state == -2 or player_id)
        enemy_players = board_state[enemy_pieces[:, 0], enemy_pieces[:, 1], 0]
        unique_enemies = np.unique(enemy_players)

        # Create a lookup table for enemy players by id
        enemy_lookup = {enemy: i for i, enemy in enumerate(unique_enemies)}             # TODO: vectorize
        enemy_mask = np.array([enemy_lookup[enemy] for enemy in enemy_players])

        # For each unique enemy player calc the valid and other mask
        is_enemy = board_state[:, :, 0] == unique_enemies[:, None, None]
        is_piece = board_state[:, :, 1] >= 0
        own_mask = np.logical_and(is_piece, is_enemy)
        other_mask = np.logical_and(is_piece, ~is_enemy)

        moves = []
        for piece, mask_id in zip(enemy_pieces, enemy_mask):
            m = self.get_moves_position(board_state, piece, own_mask[mask_id, :, :], other_mask[mask_id, :, :])
            if m is not None:
                # Add position to move
                m = np.hstack((np.tile(piece, (m.shape[0], 1)), m))
                moves.append(m)
        if moves:
            moves = np.vstack(moves)
        return moves
    
    def get_masks(self, board_state, player):
        player_territory = board_state[:, :, 0] == player
        any_pieces = board_state[:, :, 1] >= 0
        own_mask = np.logical_and(any_pieces, player_territory)
        other_mask = np.logical_and(any_pieces, ~player_territory)
        return own_mask, other_mask

    def get_moves_single(self, board_state, position):
        # First calc valid and other mask
        player = board_state[position[0], position[1], 0]
        own_mask, other_mask = self.get_masks(board_state, player)
        return self.get_moves_position(board_state, position, own_mask, other_mask)

    def get_moves_position(self, board_state, position, own_mask, other_mask):
        # Get all possible moves for the piece at position
        piece_type = board_state[position[0], position[1], 1]

        if piece_type == -1:
            return None
        elif piece_type == 0:
            return self.get_pawn_moves(board_state, position, own_mask)
        elif piece_type == 1:
            return self.get_knight_moves(position, own_mask)
        elif piece_type == 2:
            return self.get_bishop_moves(position, own_mask, other_mask)
        elif piece_type == 3:
            return self.get_rook_moves(position, own_mask, other_mask)
        elif piece_type == 4:
            return self.get_queen_moves(position, own_mask, other_mask)
        elif piece_type == 5:
            return self.get_king_moves(position, own_mask)
        else:
            raise Exception("Invalid piece type", piece_type)

    def get_pawn_moves(self, board_state, position, own_mask):
        pos1, pos2 = self.all_moves[position[0]][position[1]][0]
        # This is only valid if the square is not occupied by another piece
        valid = board_state[pos1[:, 0], pos1[:, 1], 1] == -1
        pos1 = pos1.copy()[valid, :]

        # Pawn can also capture diagonaly
        valid2 = np.logical_and(~own_mask[pos2[:, 0], pos2[:, 1]], board_state[pos2[:, 0], pos2[:, 1], 1] != -1)
        pos2 = pos2.copy()[valid2, :]
        return np.vstack((pos1, pos2))

    def get_knight_moves(self, position, own_mask):
        moves = self.all_moves[position[0]][position[1]][1]
        return moves[~own_mask[moves[:, 0], moves[:, 1]], :]
    
    def get_bishop_moves(self, position, own_mask, other_mask):
        lines, start_indices = self.all_moves[position[0]][position[1]][2]
        return self.check_lines(lines, start_indices, own_mask, other_mask)

    def get_rook_moves(self, position, own_mask, other_mask):
        lines, start_indices = self.all_moves[position[0]][position[1]][3]
        return self.check_lines(lines, start_indices, own_mask, other_mask)

    def get_queen_moves(self, position, own_mask, other_mask):
        lines, start_indices = self.all_moves[position[0]][position[1]][4]
        return self.check_lines(lines, start_indices, own_mask, other_mask)

    def get_king_moves(self, position, own_mask):
        moves = self.all_moves[position[0]][position[1]][5]
        return moves[~own_mask[moves[:, 0], moves[:, 1]], :]
    
    def check_lines(self, lines, start_indices, own_mask, other_mask):    
        # Get indices of other pieces
        # Exclude all start_indices from the other_indices
        is_other = other_mask[lines[:, 0], lines[:, 1]]
        blocked_indices = np.flatnonzero(is_other) + 1  # We can capture pieces but have to stop there.
        blocked_indices = np.setdiff1d(blocked_indices, start_indices)

        is_own = own_mask[lines[:, 0], lines[:, 1]]
        blocked_indices = np.append(blocked_indices, np.flatnonzero(is_own)) # We can't move through our own pieces
        
        if blocked_indices.shape[0] == 0:
            return lines
        
        # Add max length to blocked_indices
        blocked_indices = np.append(blocked_indices, lines.shape[0])

        # Find the indices corresponding to block
        indices = np.searchsorted(blocked_indices, start_indices, side='left')
        indices = np.clip(indices, 0, blocked_indices.size - 1)
        
        end_indices = np.where(indices != len(blocked_indices), blocked_indices[indices], lines.shape[0])

        # Apply blocking - the valid lines are between start_indices and end_indices
        valid_sections = [lines[start:end] for start, end in zip(start_indices, end_indices)]

        # Concatenate all the valid_sections into single array
        lines = np.concatenate(valid_sections)
        return lines

    def get_enemy_pieces(self, board_state, player, distance=-1):
        # Get all enemy pieces within distance of the bounding box of the player
        # Distance -1 means all
        if distance == -1:
            enemy_pieces = np.argwhere(np.logical_and(board_state[:, :, 0] >= 0, board_state[:, :, 0] != player))
            extent = np.array([[0, 0],[self.grid_height, self.grid_height]])
        else:
            bbox = np.argwhere(board_state[:, :, 0] == player)
            extent = np.array([
                np.min(bbox, axis=0) - distance,
                np.max(bbox, axis=0) + distance
            ])
            extent = np.clip(extent, 0, self.grid_height)

            # Find all enemy pieces within the bounding box
            board = board_state[extent[0, 0]:extent[1, 0], extent[0, 1]:extent[1, 1], 0]

            # Get all enemy pieces
            enemy_pieces = np.argwhere(np.logical_and(board >= 0, board != player))

            if not enemy_pieces.any():
                return None, None
            enemy_pieces += extent[0, :]
        return enemy_pieces, extent

    def precalculate_moves(self, board_state, positions):
        # Out of bounds or on water is not possible
        positions = positions.copy()
        valid = np.logical_and(np.logical_and(positions[:, 0] >= 0, positions[:, 0] < self.grid_height), np.logical_and(positions[:, 1] >= 0, positions[:, 1] < self.grid_height))
        positions = positions[valid, :]
        valid = board_state[positions[:, 0], positions[:, 1], 0] != -2
        return positions[valid, :]

    def precalculate_lines(self, board_state, lines):
        # Return lines and starting indices
        # Out of bounds or on water is not possible
        start_indices = np.cumsum(np.ones(lines.shape[0] - 1, dtype=np.int8) * lines.shape[1])

        # Flatten lines to position array
        lines = lines.copy()
        lines = lines.reshape((-1, 2))

        # Check if lines exit the board
        valid = np.logical_and(np.logical_and(lines[:, 0] >= 0, lines[:, 0] < self.grid_height), np.logical_and(lines[:, 1] >= 0, lines[:, 1] < self.grid_height))

        # For each False in valid, we want to subtract one from the start_index
        invalids = np.argwhere(~valid).flatten()

        # Add zero to start_index, add max length to invalids
        start_indices = np.insert(start_indices, 0, 0)

        if invalids.size > 0:
            invalids = np.append(invalids, lines.shape[0])

            indices = np.searchsorted(invalids, start_indices, side='left')
            indices = np.minimum(indices, invalids.shape[0] - 1)
            end_indices = np.where(indices != len(invalids), invalids[indices], lines.shape[0])

            # For each end_index take the minimum of it and the next start_index
            next_starts = np.append(start_indices[1:], lines.shape[0])
            end_indices = np.minimum(end_indices, next_starts)

            # Remove all sections [end_index[i]: start_index[i+1]] from lines array along first dimension
            delete_mask = np.hstack([np.arange(end, start) for start, end in zip(next_starts, end_indices)])
            delete_mask = delete_mask[delete_mask < lines.shape[0]]
            lines = np.delete(lines, delete_mask, axis=0)

            # Update the start_indices
            diffs = [start - next_end for start, next_end in zip(next_starts, end_indices)]
            subtr = np.cumsum(diffs)
            start_indices = start_indices[1:] - subtr[:-1]
            start_indices = np.insert(start_indices, 0, 0)

        # # Check if lines are valid, stopped by water
        invalids = np.argwhere(board_state[lines[:, 0], lines[:, 1], 0] == -2).flatten()

        if invalids.size > 0:
            invalids = np.append(invalids, lines.shape[0])

            indices = np.searchsorted(invalids, start_indices, side='left')
            indices = np.minimum(indices, invalids.shape[0] - 1)
            end_indices = np.where(indices != len(invalids), invalids[indices], lines.shape[0])

            # For each end_index take the minimum of it and the next start_index
            next_starts = np.append(start_indices[1:], lines.shape[0])
            end_indices = np.minimum(end_indices, next_starts)

            # Remove all sections [end_index[i]: start_index[i+1]] from lines array along first dimension
            delete_mask = np.hstack([np.arange(end, start) for start, end in zip(next_starts, end_indices)])
            delete_mask = delete_mask[delete_mask < lines.shape[0]]
            lines = np.delete(lines, delete_mask, axis=0)

            # Update the start_indices
            diffs = [start - next_end for start, next_end in zip(next_starts, end_indices)]
            subtr = np.cumsum(diffs)
            start_indices = start_indices[1:] - subtr[:-1]
            start_indices = np.insert(start_indices, 0, 0)
        
        start_indices = np.unique(start_indices)
        if start_indices[-1] == lines.shape[0]:
            start_indices = start_indices[:-1]
            
        return lines, start_indices

    def execute_move(self, board_state, move, inplace=True):
        # Execute a move on the board state
        # Move is a tuple (x, y, x2, y2) starting position and end position
        if not inplace:
            board_state = board_state.copy()
        x, y, x2, y2 = move
        board_state[x2, y2, :] = board_state[x, y, :]
        board_state[x, y, 1] = -1
        return board_state

    def evaluate_board(self, board_state, player):
        # Evaluate the board for a player
        # Territory score
        is_player = (board_state[:, :, 0] == player)
        territory_score = np.count_nonzero(is_player)

        # Piece score
        own_pieces = np.logical_and(board_state[:, :, 1] >= 0, is_player)
        piece_score = np.sum(piece_values[board_state[own_pieces, 1]])

        # Enemy piece score (normalized to viewing area)
        enemy_pieces, extent = self.get_enemy_pieces(board_state, player)
        enemy_piece_score = 0
        if enemy_pieces is not None:
            enemy_piece_score = np.sum(piece_values[board_state[enemy_pieces[:, 0], enemy_pieces[:, 1], 1]])
            enemy_piece_score /= (extent[1, 0] - extent[0, 0]) * (extent[1, 1] - extent[0, 1])

        return territory_score + piece_score - enemy_piece_score