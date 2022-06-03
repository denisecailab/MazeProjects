import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

from util import check_attrs


def overlay_footprints(ScrollObj):
    attrs = ["footprints"]
    check_attrs(ScrollObj, attrs)

    footprints = ScrollObj.footprints
    dims = footprints[0][0].shape
    overlay = np.zeros((dims[0], dims[1], 3))

    for channel, footprints_this_day in enumerate(footprints):
        footprint = footprints_this_day[ScrollObj.current_position]
        overlay[:, :, channel] = footprint / np.max(footprint)

    ax = ScrollObj.ax
    ax.imshow(overlay, origin="lower")

    ScrollObj.last_position = footprints[0].shape[0]


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, label=None):
    """
    Line show_plot with error bars except the error bars are filled in
    rather than the monstrosity from matplotlib.

    :parameters
    ---
    x: array-like
        x-axis values.

    y: array-like, same length as x
        y-axis values.

    yerr: array-like, same length as x and y
        Error around the y values.
    """
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def jitter_x(arr, jitter=0.05):
    jittered = arr + np.random.randn(len(arr)) * jitter

    return jittered

def beautify_ax(ax):
    ax.tick_params(right="off",top="off",length = 4, width = 1, direction = "out")
    [ax.spines[side].set_visible(False) for side in ['top', 'right']]
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    for line in ["left","bottom"]:
        ax.spines[line].set_linewidth(2)
        ax.spines[line].set_position(("outward",10))

    return ax

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def plot_xy_line(ax):
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
