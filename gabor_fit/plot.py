import numpy as np
import matplotlib.pyplot as plt

from utils import tile_raster_images

def plot_gabors(gabors):
    n, m = gabors.shape
    side = int(np.round(np.sqrt(m)))
    w = int(np.sqrt(n))
    if w**2 < n:
        h = w + 1
    else:
        h = w
    img = tile_raster_images(gabors, (side, side), (w, h), (2, 2))
    plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.show()
