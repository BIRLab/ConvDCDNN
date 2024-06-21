import numpy as np
from matplotlib import colormaps
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from scipy.interpolate import RBFInterpolator


def interpolate_channel_data(coords: np.ndarray, channel_data: np.ndarray, resolution=67):
    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, resolution), np.linspace(-0.5, 0.5, resolution))
    mask = xx ** 2 + yy ** 2 >= 0.5 ** 2
    xx[mask] = np.nan
    yy[mask] = np.nan
    zinterp = RBFInterpolator(coords, channel_data)
    return zinterp(np.stack((xx.flatten(), yy.flatten()), axis=1)).reshape((resolution, resolution))


def plot_head_edge(_fig: Figure, ax: Axes, linewidth):
    # plot ear nose
    ear_x = np.array([.497, .510, .518, .5299, .5419, .54, .547, .532, .510, .489])
    ear_y = np.array([.0555, .0775, .0783, .0746, .0555, -.0055, -.0932, -.1313, -.1384, -.1199])
    circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot([0.09, 0, -0.09], [0.496, 0.575, 0.496], 'k', linewidth=linewidth)
    ax.plot(ear_x, ear_y, 'k', linewidth=linewidth)
    ax.plot(-ear_x, ear_y, 'k', linewidth=linewidth)
    ax.plot(0.5 * np.cos(circle), 0.5 * np.sin(circle), 'k', linewidth=linewidth)


def plot_top(fig: Figure, ax: Axes, coords: np.ndarray, channels: list[str], channel_data: np.ndarray, resolution=67, fontsize=6, colorbar=True, linewidth=1.0, circle_linewidth=0.5):
    # plot channel value
    zz = interpolate_channel_data(coords, channel_data, resolution)
    im = ax.imshow(zz, extent=(-0.502, 0.502, -0.502, 0.502), cmap=colormaps["jet"], origin="lower")
    if colorbar:
        fig.colorbar(im)

    # plot head edge
    plot_head_edge(fig, ax, linewidth)

    # plot channel circle
    for i in range(64):
        ax.add_patch(Circle((coords[i, 0], coords[i, 1]), radius=0.03, fill=False, linewidth=circle_linewidth))
        ax.annotate(channels[i], xy=(coords[i, 0], coords[i, 1]), fontsize=fontsize, va="center", ha="center")

    # hide ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return im
