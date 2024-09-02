import matplotlib.pyplot as plt
import numpy as np


def cplot(image):
    plt.imshow(image, cmap="binary", vmin=0, vmax=1)


def reflect_matrix():
    return np.array([[-1, 0], [0, 1]])


def rot_matrix(theta):
    return np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )


def line_indices(x1, y1, x2, y2):
    n_pix = np.max([abs(x1 - x2), abs(y1 - y2)]) + 1
    x_ind = (
        np.linspace([y1, x1], [y2, x2], n_pix, dtype=float).round().astype(int)
    )
    return x_ind.transpose()
