# A python class to handle a single instance of a macromate game
# All computing is performed using numpy

# Board is 64 x 64 squares, origin at top left
from random import shuffle
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import cProfile, pstats

from utils import PIECE_NAMES, Utils

IMG_FOLDER = "data/img/"

class MacroMateEngine(object):
    
    def __init__(self, num_players=6):
        self.num_players = num_players

        # Load land / water map as a numpy array
        self.land_grid = 1 - np.load('map/data/land_grid.npy')

        self.map_shape = self.land_grid.shape
        self.grid_height = self.map_shape[0]

        # Inland grid is where you can place a king (all surounding squares are land and on the map)
        self.inland_grid = np.zeros(self.map_shape, dtype=int)
        for x in range(2, self.grid_height - 2):
            for y in range(2, self.grid_height - 2):
                if self.land_grid[x, y] == 1:
                    if np.min(self.land_grid[x-1:x+2, y-1:y+2]) == 1:
                        self.inland_grid[x, y] = 1

        # IMAGES FOR PIECES
        self.piece_images = {}
        for i, img in enumerate(["pawn", "knight", "bishop", "rook", "queen", "king"]):
            img = plt.imread(IMG_FOLDER + img + ".png")
            img = np.flip(img, axis=0)
            self.piece_images[i] = img

        # PLAYER COLORS for all 8 players (nice color scheme)
        player_colors = [
            "#1f77b4ff", # blue
            "#ff7f0eff", # orange
            "#2ca02cff", # green
            "#d62728ff", # red
            "#9467bdff", # purple
            "#8c564bff", # brown
            "#e377c2ff", # pink
            "#227ffeff"  # light blue
        ]
        self.player_colors = [colors.to_rgba(c) for c in player_colors]
        self.transparent_color = colors.to_rgba("#00000000")

        # render all images in the player colors (for the pieces)
        self.piece_player_images = []
        for player_i in range(self.num_players):
            pl_imgs = []
            for piece_i in range(6):
                img = self.piece_images[piece_i].copy()

                # Replace all white pixels with the player color
                img[img[:, :, 0] == 1] = self.player_colors[player_i]

                pl_imgs.append(img)
            self.piece_player_images.append(pl_imgs)

        self.move_scatter = None
        self.position_selected = None
        self.ui_possible_moves = None
        
        self.reset_board_state()
        self.utils = Utils(self.board_state)
        
    def reset_board_state(self):
        # BOARD STATE
        # Territory ownership as a 2D numpy array (height, height), this is both current and last occupied squares
        # -2 = water, -1 = empty, 0 = player 0, 1 = player 1, 2 = player 2, ...

        # Pieces as a 2D numpy array (height, height)
        # -1 = empty, 0 = pawn, 1 = knight, 2 = bishop, 3 = rook, 4 = queen, 5 = king
        self.board_state = np.ones((self.grid_height, self.grid_height, 2), dtype=int) * -1 # 2 for territory ownership and pieces 
        self.board_state[:, :, 0] = self.land_grid - 2 # Set water to -2, normal land to -1

    def get_random_king_position(self):
        while True:
            x = np.random.randint(self.grid_height)
            y = np.random.randint(self.grid_height)
            if self.inland_grid[x, y] == 1: 
                if np.max(self.board_state[x-2:x+3, y-2:y+3, 1]) == -1:
                    return x, y

    def setup_random_starts(self, seed=-1):
        # Randomly place all players on the board with a starting setup:
        # King, surounded by 8 pawns
        self.reset_board_state()
        
        if seed != -1:
            np.random.seed(seed)
        
        for player in range(self.num_players):
            # Place king and pawns
            x, y = self.get_random_king_position()
            self.board_state[x-1:x+2, y-1:y+2, :] = np.array([player, 0])
            self.board_state[x, y, 1] = 5

            # Everybody gets one of each of the other pieces in the corners
            corners = np.array([[x-1, y-1], [x-1, y+1], [x+1, y-1], [x+1, y+1]])
            pieces = np.array([1, 2, 3, 4])
            np.random.shuffle(pieces)
            self.board_state[corners[:, 0], corners[:, 1], 1] = pieces

    def get_best_move(self, player):
        # Get the best move for a player
        return self.utils.search_tree(self.board_state, player)

    def simulate_one_round(self):
        # Simulate one round of the game
        before = time.time()
        for player in range(1, self.num_players):
            move = self.get_best_move(player)
            if move is not None:
                self.execute_move(move)

        duration = round(time.time() - before, 1)
        print(f"Simulated {self.num_players-1} bots in {duration} seconds")
    
    def profile_one_round(self):
        # Profile simulating one round -> call_graph.png
        def profile_round_raw():
            self.simulate_one_round()
        
        cProfile.runctx('profile_round_raw()', globals(), locals(), 'profile_stats')
        subprocess.run(['gprof2dot', '-f', 'pstats', 'profile_stats', '-o', 'call_graph.dot'])
        subprocess.run(['dot', '-Tpng', 'call_graph.dot', '-o', 'call_graph.png'])

    
    def execute_move(self, move):
        self.board_state = self.utils.execute_move(self.board_state, move)

    def execute_human_move(self, move, profile=False):
        self.execute_move(move)
        self.position_selected = None
        self.ui_possible_moves = None

        if profile:
            self.profile_one_round()
        else:
            self.simulate_one_round()
        
        self.update_board()

    def onclick(self, event):
        y = int(event.xdata + 0.5)
        x = int(event.ydata + 0.5)
        bs = self.board_state[x, y, :]
        position = np.array([x, y])

        do_human_move = self.ui_possible_moves is not None and self.position_selected is not None

        if do_human_move:
            # Search for the move in the possible moves numpy
            for move in self.ui_possible_moves:
                if move[0] == x and move[1] == y:
                    print("Moved!")
                    move = [self.position_selected[0], self.position_selected[1], x, y]
                    self.execute_human_move(move)
                    return
            else:
                self.position_selected = None
                self.ui_possible_moves = None

        if bs[1] != -1:
            # if no piece is selected, we might select this one.
            if self.position_selected is None:
                if bs[0] == 0:
                    self.position_selected = position.copy()
                else:
                    self.position_selected = None
                    self.ui_possible_moves = None

                # Get all moves for this position
                self.ui_possible_moves = self.utils.get_moves_single(self.board_state, position)
                self.move_scatter = plt.scatter(self.ui_possible_moves[:, 1], self.ui_possible_moves[:, 0], c="green", s=50)
                plt.draw()

    def update_board(self):

        # Create a new figure only if there is no other plot open
        created_figure = False
        if len(plt.get_fignums()) == 0:
            plt.figure(figsize=(12, 12))
            created_figure = True
        else:
            # Clear the whole plot
            plt.clf()

        # Create the chess pattern on land
        x = np.arange(self.land_grid.shape[1])
        y = np.arange(self.land_grid.shape[0])
        xv, yv = np.meshgrid(x, y)
        chess_pattern = (xv + yv) % 2  # gives a grid with values 1 where the sum of the x and y coordinates is even, and 0 where it's odd

        # Set the alpha of the chess pattern to 0.2 on water
        plt.imshow(chess_pattern, cmap='gray', alpha=0.2)
        plt.imshow(self.land_grid, cmap='Greens', alpha=0.5)

        # Plot the territory ownership but only if >= 0 (not water or unowned)
        territories = self.board_state[:, :, 0]

        # Convert to a rgba array with the player colors, -1 and -2 use transparent color
        rgba = np.zeros((self.grid_height, self.grid_height, 4))
        for player_i in range(self.num_players):
            rgba[territories == player_i] = self.player_colors[player_i]
        plt.imshow(rgba, alpha=0.3)

        # Plot all the pieces
        for x in range(self.grid_height):
            for y in range(self.grid_height):
                piece_type = self.board_state[x, y, 1]
                if piece_type != -1:
                    player = self.board_state[x, y, 0]
                    # TODO: Remove axis flip
                    extent = [y-0.5, y+0.5, x-0.5, x+0.5]
                    plt.imshow(self.piece_player_images[player][piece_type], extent=extent)
        

        # Click handler on plot
        cid = plt.gcf().canvas.mpl_connect('button_press_event', self.onclick)
        
        # No padding in figure
        plt.axis('off')
        plt.tight_layout(pad=2)

        # Adjust plot limits to chess_pattern
        w = self.grid_height - 0.5
        plt.xlim(-0.5, w)
        plt.ylim(w, -0.5)

        # Show plot only if no other plots are open
        if created_figure:
            plt.show()
        else:
            plt.draw()

if __name__ == "__main__":
    engine = MacroMateEngine()

    engine.setup_random_starts(seed=1)

    # engine.simulate_one_round()
    engine.update_board()



    # Profile 1000 rounds
    # def profile_1000_rounds():
    #     for _ in range(1):
    #         engine.simulate_one_round()
    # cProfile.run('profile_1000_rounds()', 'profile_stats')
    # # stats = pstats.Stats('profile_stats')
    # # stats.sort_stats('tottime')
    # # stats.print_stats()
    # # Convert profiling data to call graph
    # subprocess.run(['gprof2dot', '-f', 'pstats', 'profile_stats', '-o', 'call_graph.dot'])

    # # Convert DOT file to PNG image
    # subprocess.run(['dot', '-Tpng', 'call_graph.dot', '-o', 'call_graph.png'])

    # engine.show_board()