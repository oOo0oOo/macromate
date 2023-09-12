
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs


X_LIMITS = (-10, 23)  # (-10, 37)
Y_CENTER = 45
GRID_HEIGHT = 32

X_DIFF = X_LIMITS[1] - X_LIMITS[0]

Y_LIMITS = (Y_CENTER - X_DIFF / 2, Y_CENTER + X_DIFF / 2)

GRID_SIZE = X_DIFF / GRID_HEIGHT

# GRID_SIZE = 0.6 # 0.55
grid_shape = np.array((GRID_HEIGHT, GRID_HEIGHT))
# grid_shape = np.array((int((X_LIMITS[1] - X_LIMITS[0]) / GRID_SIZE), int((Y_LIMITS[1] - Y_LIMITS[0]) / GRID_SIZE)))
print("Grid shape:", grid_shape)

# robinson = ccrs.Robinson().proj4_init

# Plot the world map 1 bit land / water
def save_world_map():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Export the world map as a high resolution png
    fig, ax = plt.subplots(figsize=(30, 20))
    ax.axis('off')
    
    # Change projection to robinson
    # world.to_crs(robinson).plot(ax=ax, color='black')

    # Center projection on europe
    # ax.set_extent([X_LIMITS[0], X_LIMITS[1], Y_LIMITS[0], Y_LIMITS[1]], crs=ccrs.PlateCarree())

    world.boundary.plot(ax=ax, color='black', linewidth=0.3)
    world.plot(ax=ax, color='black')

    # ALso plot population density
    # world.plot(ax=ax, column='pop_est', cmap='OrRd', legend=True)
    
    plt.xlim(X_LIMITS[0], X_LIMITS[1])
    plt.ylim(Y_LIMITS[0], Y_LIMITS[1])

    plt.savefig('data/world_map.png', dpi=300, bbox_inches='tight', pad_inches=0)

    # Delete the plot
    plt.close()


def save_chess_grid():
    # Overlay chess grid onto the world map
    # Use world_map to decide whether a grid is land or water
    # Save the chess grid as a numpy array
    world_map = plt.imread('data/world_map.png')
    world_map = world_map[:, :, 0]

    img_scale = grid_shape[0] / world_map.shape[1]
    grid_width_img = int(GRID_SIZE / img_scale)

    # Create a chess grid
    land_grid = np.zeros(grid_shape, dtype=bool)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # Cut square from world map
            x = int(i / img_scale)
            y = int(j / img_scale)
            snippet = world_map[y:y+grid_width_img, x:x+grid_width_img]
            water = np.sum(snippet) / (grid_width_img ** 2) > 0.7
            land_grid[i, j] = water

    land_grid = land_grid.T

    # Save the land grid
    np.save('data/land_grid.npy', land_grid)

    x = np.arange(land_grid.shape[1])
    y = np.arange(land_grid.shape[0])
    xv, yv = np.meshgrid(x, y)
    chess_pattern = (xv + yv) % 2  # gives a grid with values 1 where the sum of the x and y coordinates is even, and 0 where it's odd

    # Plot the result
    plt.figure(figsize=(30, 20))
    plt.imshow(chess_pattern, cmap='gray', alpha=1)
    plt.imshow(1 - land_grid, cmap='tab20c', alpha=0.8)


    # Onclick: flip the clicked square (land/water) and update the plot, save the land grid
    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return

        y = int(round(event.xdata - 0.5))
        x = int(round(event.ydata - 0.5))

        land_grid[x, y] = not land_grid[x, y]

        plt.clf()
        plt.imshow(chess_pattern, cmap='gray', alpha=1)
        plt.imshow(1 - land_grid, cmap='tab20c', alpha=0.8)
        plt.draw()

        np.save('data/land_grid.npy', land_grid)
    
    plt.connect('button_press_event', onclick)
    plt.show()


if __name__ == '__main__':
    save_world_map()
    save_chess_grid()