import numpy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import zscore
import os
from tqdm import tqdm
from CaImaging import util
from CaImaging.Behavior import spatial_bin


def plot_assembly(
    pattern,
    activation,
    spike_times,
    sort_by_contribution=True,
    order=None,
    ax=None,
    frames=None,
    activation_color="m",
    spike_colors='k',
):
    """
    Plots single assemblies. This plot contains the activation profile (activation over time)
    in addition to a raster in the background. There is an option to sort neurons on the raster
    according to their contribution.

    :parameters
    ---
    pattern: array-like
        ICA weights from a single assembly.

    activation: array-like
        Activation profile of a single assembly.

    spike_times: list of lists
        Containing indices of calcium transients.

    sort_by_contribution: boolean
        Whether to sort spikes based on contribution.

    order: None or array-like, same size as pattern.
        Order in which to sort.

    ax: Axes object
        To plot on top of.

    :returns
    ---
    activation_ax: Axes object
        Axes containing activation plot.

    spikes_ax: Axes object
        Axes containing raster plot.

    """
    # Define axes.
    if ax is None:
        fig, ax = plt.subplots()
    activation_ax = ax
    spikes_ax = activation_ax.twinx()

    if type(spike_colors) is str:
        spike_colors = np.asarray([spike_colors for i in range(len(spike_times))])

    # Get sort order for neurons based on contribution to assembly.
    if sort_by_contribution and order is None:
        order = np.argsort(np.abs(pattern))
        sorted_spike_times = [spike_times[n] for n in order]
        spike_colors = spike_colors[order]
    elif not sort_by_contribution and order is not None:
        sorted_spike_times = [spike_times[n] for n in order]
        spike_colors = spike_colors[order]
    else:
        sorted_spike_times = spike_times

    if frames is None:
        frames = range(len(activation))

    activation_ax.plot(frames, activation, linewidth=2, color=activation_color)
    xlims = list(activation_ax.get_xlim())
    activation_ax.set_xticks(xlims)
    activation_ax.set_xticklabels([0, 1800])
    activation_ax.set_ylabel(
        "Activation strength (a.u.)", color=activation_color, fontsize=22
    )
    activation_ax.set_xlabel("Time (s)", fontsize=22)
    spikes_ax.eventplot(sorted_spike_times, color=spike_colors, alpha=0.2, rasterized=True)
    spikes_ax.set_ylabel("Neurons", rotation=-90, fontsize=22)
    spikes_ax.set_yticks([0, len(sorted_spike_times)])

    return activation_ax, spikes_ax


def write_assembly_triggered_movie(
    activation,
    frame_numbers,
    behavior_movie,
    fpath=None,
    threshold=2.58,
    trials=None,
):
    z_activation = zscore(activation)
    inds = np.where(z_activation > threshold)[0]
    above_threshold_frames = frame_numbers[inds]

    grouped_frames = util.cluster(above_threshold_frames, 30)
    grouped_inds = util.cluster(inds, 30)

    compressionCodec = "XVID"
    codec = cv2.VideoWriter_fourcc(*compressionCodec)
    cap = cv2.VideoCapture(behavior_movie)
    ret, frame = cap.read()
    rows, cols = int(cap.get(4)), int(cap.get(3))

    if fpath is None:
        folder = os.path.split(behavior_movie)[0]
        fpath = os.path.join(folder, "Assembly activity.avi")

    writeFile = cv2.VideoWriter(fpath, codec, 15, (cols, rows), isColor=True)
    print(f"Writing {fpath}")
    for i, (chunked_frames, chunked_inds) in enumerate(
        zip(grouped_frames, grouped_inds)
    ):
        for frame_number in chunked_frames:
            cap.set(1, frame_number)
            ret, frame = cap.read()
            writeFile.write(frame)

        blank_frame = np.zeros_like(frame)
        text = f"Activation #{i}"

        cv2.putText(
            blank_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        if trials is not None:
            trial_text = f"Lap # {trials[chunked_inds[0]]}"
            cv2.putText(
                blank_frame,
                trial_text,
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        for i in range(15):
            writeFile.write(blank_frame)

    writeFile.release()
    cap.release()


def find_members(patterns, filter_method="sd", thresh=2):
    """
    Find the members of an ensemble defined by high weights exceeding some threshold.

    :parameters
    ---
    patterns: (assembly, neuron) array
        Patterns to find members from.

    filter_method: str ('sd' or 'z')
        Determines the method for finding members. If 'sd', looks for neurons above x standard deviations above or below
        the mean, determined by thresh.

    thresh: float
        Threshold for calling a neuron an ensemble member. If filter_method is 'sd', denotes the number of standard
        deviations above or below the mean. If filter method is 'z', denotes the z-score (positive or negative).

    :return
    ---

    """
    if patterns.ndim == 1:
        patterns = patterns[np.newaxis, :]
    n_patterns, n_neurons = patterns.shape

    member_candidates = np.zeros((2, *patterns.shape))
    if filter_method == "sd":
        # Take the mean and standard deviation to find high neuron weights.
        pattern_mean = np.mean(patterns, axis=1)
        pattern_stds = np.std(patterns, axis=1)
        member_candidates[0] = (
            patterns > np.tile(pattern_mean + thresh * pattern_stds, (n_neurons, 1)).T
        )
        member_candidates[1] = (
            patterns < np.tile(pattern_mean - thresh * pattern_stds, (n_neurons, 1)).T
        )

        # Determine if the weight vectors have negative or positive values for ensemble members.
        # Do this by counting number of highly contributing neurons in both negative and positive and finding which
        # (positive or negative) are more plentiful.
        signs = np.argmax(np.sum(member_candidates, axis=2), axis=0)

        # For each assembly, select the positive or negative signed weights.
        bool_members = np.zeros_like(patterns)
        corrected_patterns = patterns.copy()
        for assembly_i, assembly_sign in enumerate(signs):
            bool_members[assembly_i] = member_candidates[assembly_sign, assembly_i]

            # assembly_sign == 1 means the negative weights contain highly-contributing neurons.
            # In this case, negate the pattern and store it.
            if assembly_sign == 1:
                corrected_patterns[assembly_i] = -corrected_patterns[assembly_i]

        # For each ensemble, get the index of bool_members.
        member_idx_ = [np.where(members)[0] for members in bool_members]

    elif filter_method == "z":
        raise ValueError("z method not done yet")
        # bool_members = patterns > thresh
        # member_idx = [np.where(members)[0] for members in bool_members]

    else:
        raise ValueError("Unaccepted filter_method.")

    # Sort neurons by their weight.
    member_idx = []
    sort_orders = np.argsort(corrected_patterns, axis=1)
    for i, order in enumerate(sort_orders):
        corrected_patterns[i] = corrected_patterns[i, order]
        member_idx.append([neuron for neuron in order if neuron in member_idx_[i]])

    # If there was only one pattern in the input, just take the first row/list entry.
    if n_patterns == 1:
        member_idx = member_idx[0]
        corrected_patterns = np.squeeze(corrected_patterns)

    return bool_members, member_idx, corrected_patterns


def find_memberships(patterns, filter_method="sd", thresh=2):
    """
    Rather than finding the members for each ensemble, do the inverse -- find which ensemble each neuron belongs to.

    :param patterns:
    :param filter_method:
    :param thresh:
    :return:
    """
    member_list = find_members(patterns, filter_method=filter_method, thresh=thresh)[1]

    memberships = []
    n_neurons = patterns.shape[1]

    for neuron in range(n_neurons):
        memberships.append([])
        for i, ensemble in enumerate(member_list):
            if neuron in ensemble:
                memberships[neuron].append(i)

    return memberships


def spatial_bin_ensemble_activations(
    activations,
    lin_position,
    occupancy_normalization,
    spatial_bin_size_radians=0.05,
    do_zscore=True,
):

    # Bin all the assembly activations in space.
    ensemble_fields = []
    if do_zscore:
        activations = zscore(activations, axis=1)
    for assembly in activations:
        assembly_field = spatial_bin(
            lin_position,
            np.zeros_like(lin_position),
            bin_size_cm=spatial_bin_size_radians,
            show_plot=False,
            weights=assembly,
            one_dim=True,
        )[0]
        ensemble_fields.append(assembly_field / occupancy_normalization)
    ensemble_fields = np.vstack(ensemble_fields)

    return ensemble_fields


def plot_pattern(
    pattern, ax=None, color="k", alpha=1, linewidth=0.5, markersize=5, order=None
):
    if ax is None:
        fig, ax = plt.subplots()

    n_neurons = len(pattern)
    if order is None:
        order = range(n_neurons)

    markerline, stemlines, baseline = ax.stem(
        range(n_neurons), pattern[order], color, markerfmt="o", basefmt=" "
    )
    plt.setp(markerline, color=color, alpha=alpha)
    plt.setp(stemlines, color=color, alpha=alpha)
    plt.setp(stemlines, "linewidth", linewidth)
    plt.setp(markerline, "markersize", markersize)

    ax.set_ylabel("Weight [a.u.]", fontsize=22)
    ax.set_xlabel("Neurons", fontsize=22)

    return ax
