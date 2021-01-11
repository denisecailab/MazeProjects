from CircleTrack.BehaviorFunctions import BehaviorSession, spiral_plot
import matplotlib.pyplot as plt
import numpy as np
from CaImaging.util import sync_data, nan_array
from CaImaging.Miniscope import threshold_S
from util import Session_Metadata
from CircleTrack.BehaviorFunctions import linearize_trajectory
from CaImaging.util import ScrollPlot
from CircleTrack.plotting import plot_spiral, plot_raster
from CaImaging.PlaceFields import PlaceFields
from CaImaging.Behavior import spatial_bin
import holoviews as hv
import os
import pickle as pkl

hv.extension("bokeh")
from bokeh.plotting import show
from itertools import product
from scipy.stats import spearmanr, zscore
from matplotlib import colors


class CalciumSession:
    def __init__(
        self,
        session_folder,
        spatial_bin_size_radians=0.05,
        S_std_thresh=1,
        circle_radius=38.1,
        data_fname="SyncedData.pkl",
        placefield_fname="Placefields.pkl",
        placefield_trials_fname="PlacefieldTrials.pkl",
        overwrite_synced_data=True,
        overwrite_placefields=False,
        overwrite_placefield_trials=False,
    ):
        """
        Single session analyses and plots for miniscope data.

        :parameter
        session_folder: str
            Session folder.
        """
        # Get the behavioral data.
        self.folder = session_folder
        self.spatial_bin_size = spatial_bin_size_radians
        self.S_std_thresh = S_std_thresh

        fpath = os.path.join(self.folder, data_fname)
        try:
            if overwrite_synced_data:
                print(f"Overwriting {fpath}.")
                raise Exception
            with open(fpath, "rb") as file:
                self.data = pkl.load(file)

            self.minian_path = self.data["behavior"].paths["minian"]
        except:
            self.data = {"behavior": BehaviorSession(self.folder)}

            # Get paths
            self.minian_path = self.data["behavior"].paths["minian"]
            timestamp_paths = self.data["behavior"].paths["timestamps"]

            # Combine behavioral and calcium imaging data.
            self.data["behavior"].behavior_df, self.data["imaging"], _ = sync_data(
                self.data["behavior"].behavior_df, self.minian_path, timestamp_paths
            )

            # Redo trial counting to account in case some frames got cut
            # from behavior (e.g. because a miniscope video got truncated).
            self.data["behavior"].ntrials = max(
                self.data["behavior"].behavior_df["trials"] + 1
            )

            with open(fpath, "wb") as file:
                pkl.dump(self.data, file)

        self.data["imaging"]["S_binary"] = threshold_S(
            self.data["imaging"]["S"], S_std_thresh
        )

        # Get number of neurons.
        self.n_neurons = self.data["imaging"]["C"].shape[0]

        # Get place fields.
        fpath = os.path.join(self.folder, placefield_fname)
        self.spatial = dict()
        try:
            if overwrite_placefields:
                print(f"Overwriting {fpath}.")
                raise Exception

            with open(fpath, "rb") as file:
                self.spatial["placefield_class"] = pkl.load(file)

            parameters_match = [
                self.spatial["placefield_class"].bin_size == spatial_bin_size_radians,
                self.spatial["placefield_class"].circle_radius == circle_radius,
            ]

            if not all(parameters_match):
                print("A placefield parameter does not match saved data, rerunning.")
                raise Exception

        except:
            self.spatial["placefield_class"] = PlaceFields(
                np.asarray(self.data["behavior"].behavior_df["t"]),
                np.asarray(self.data["behavior"].behavior_df["lin_position"]),
                np.zeros_like(self.data["behavior"].behavior_df["lin_position"]),
                self.data["imaging"]["S"],
                bin_size=self.spatial_bin_size,
                circular=True,
                fps=self.data["behavior"].fps,
                circle_radius=circle_radius,
                shuffle_test=True
            )

            with open(fpath, "wb") as file:
                pkl.dump(self.spatial["placefield_class"], file)

        # Get spatial activity by trial.
        fpath = os.path.join(session_folder, placefield_trials_fname)
        try:
            if overwrite_placefield_trials:
                print(f"Overwriting {fpath}")
                raise Exception

            with open(fpath, "rb") as file:
                (self.spatial["trial_fields"], self.spatial["occupancy"]) = pkl.load(
                    file
                )
        except:
            (
                self.spatial["trial_fields"],
                self.spatial["occupancy"],
            ) = self.spatial_activity_by_trial()

            with open(fpath, "wb") as file:
                pkl.dump(
                    (self.spatial["trial_fields"], self.spatial["occupancy"]), file
                )

    def plot_spiral_spikes(self, first_neuron=0):
        """
        Plot where on the maze a neuron spikes, starting with
        first_neuron. Scroll through the rest.

        :parameter
        first_neuron: int
            Neuron to start plotting.

        S_thresh: float
            Number of standard deviations above the mean to consider
            a spike.
        """
        # Get spiking activity and time vector.
        spikes = self.data["imaging"]["S_binary"]
        t = np.asarray(self.data["behavior"].behavior_df["frame"])

        # Linearize position.
        lin_position = np.asarray(
            linearize_trajectory(self.data["behavior"].behavior_df)[0]
        )

        # Do the show_plot.
        cell_number_labels = [f"Cell #{n}" for n, _ in enumerate(spikes)]
        self.spiral_spatial_plot = ScrollPlot(
            plot_spiral,
            current_position=first_neuron,
            t=t,
            lin_position=lin_position,
            markers=spikes,
            marker_legend="Spikes",
            subplot_kw={"projection": "polar"},
            titles=cell_number_labels,
        )

    def scrollplot_rasters(self, neurons=None, binary=True):
        if neurons is None:
            neurons = range(self.n_neurons)

        rasters = self.spatial['trial_fields'][neurons]
        if binary:
            rasters = rasters > 0

        tuning_curves = self.spatial['placefield_class'].pfs[neurons]

        cell_number_labels = [f"Cell #{n}" for n in neurons]
        self.raster_plot = ScrollPlot(plot_raster,
                                      current_position=neurons[0],
                                      nrows=2,
                                      rasters=rasters,
                                      tuning_curves=tuning_curves,
                                      binary=binary,
                                      titles=cell_number_labels,
                                      figsize=(3,6.5))

        pass


    def spatial_activity_by_trial(self):
        """
        Plot activity trial by trial, binned in linearized space.

        :parameter
        ---
        bin_size_radians: float
            Spatial bin size in radians.

        """
        # Get linearized position. We'll also need a dummy variable
        # for the binning function because it usually takes 2d.
        behavior_df = self.data["behavior"].behavior_df
        lin_position = np.asarray(behavior_df["lin_position"])
        filler = np.zeros_like(lin_position)
        bin_edges = spatial_bin(
            lin_position, filler, bin_size_cm=self.spatial_bin_size, one_dim=True
        )[1]

        # Threshold S matrix here.
        S = threshold_S(self.data["imaging"]["S"], std_thresh=1)

        # For each trial, spatial bin position weighted by S.
        occ_map_by_trial = []
        fields = nan_array(
            (self.n_neurons, self.data["behavior"].ntrials, len(bin_edges) - 1)
        )
        for trial_number in range(self.data["behavior"].ntrials):
            time_bins = behavior_df["trials"] == trial_number
            positions_this_trial = behavior_df.loc[time_bins, "lin_position"]
            filler = np.zeros_like(positions_this_trial)

            # Get occupancy this trial.
            occupancy = spatial_bin(
                positions_this_trial, filler, bins=bin_edges, one_dim=True
            )[0]

            # Weight position by activity of each neuron.
            for n, neuron in enumerate(S):
                spiking = neuron[time_bins]
                fields[n, trial_number, :] = spatial_bin(
                    positions_this_trial,
                    filler,
                    bins=bin_edges,
                    one_dim=True,
                    weights=spiking,
                )[0]

            occ_map_by_trial.append(occupancy)
        occ_map_by_trial = np.vstack(occ_map_by_trial)

        return fields, occ_map_by_trial

    def viz_spatial_trial_activity(self, neurons=range(10), preserve_neuron_idx=True):
        """
        Visualize single cell activity binned in spatial and separated
        by trials.

        :parameters
        ---
        bin_size_radians: float
            Spatial bin size in radians.

        neurons: array-like of ints
            List of neuron indices.

        preserve_neuron_idx: boolean
            In the output dict, viz_fields, this flag either keeps the
            neuron index or reassigns all neurons to new dict keys
            starting from 0. The latter option is useful for neurons
            registered across days and serves as a global index for
            Holomap to access.
        """
        fields = self.spatial_activity_by_trial()[0]

        if preserve_neuron_idx:
            viz_fields = {n: hv.Image(fields[n] > 0).opts(cmap="gray") for n in neurons}
        else:
            viz_fields = {
                i: hv.Image(fields[n] > 0).opts(cmap="gray")
                for i, n in enumerate(neurons)
            }

        return viz_fields

    def correlate_spatial_PVs_by_trial(self, show_plot=True):
        """
        Correlate trial pairs of spatial ratemaps.

        :return
        corr_matrix: (trial, trial) np.array
            Correlation matrix.
        """
        # Change the axes of the fields from (cell, trial, spatial bin)
        # to (trial, cell, spatial bin).
        # fields = np.asarray([field / self.spatial['occupancy'] for field in self.spatial['fields']])
        fields = self.spatial["trial_fields"]
        fields_by_trial = np.rollaxis(fields, 1)

        # Compute spearman correlation coefficient.
        corr_matrix = np.zeros((fields_by_trial.shape[0], fields_by_trial.shape[0]))
        for i, (a, b) in enumerate(product(fields_by_trial, repeat=2)):
            idx = np.unravel_index(i, corr_matrix.shape)
            corr_matrix[idx] = spearmanr(a.flatten(), b.flatten(), nan_policy="omit")[0]

        # Fill diagonal with 0s.
        np.fill_diagonal(corr_matrix, 0)
        # offset = colors.DivergingNorm(vcenter=0)

        if show_plot:
            fig, ax = plt.subplots()
            ax.imshow(corr_matrix, cmap="bwr", vmin=-1, vmax=1)

        return corr_matrix


if __name__ == "__main__":
    folder = (
        r"Z:\Will\Drift\Data\Castor_Scope05\09_10_2020_CircleTrackReversal1\15_49_24"
    )
    S = CalciumSession(folder)
    pvals = S.spatial['placefield_class'].pvals
    S.scrollplot_rasters(neurons=np.where(np.asarray(pvals) < 0.01)[0],
                         binary=True)
    # S.spatial_activity_by_trial(0.1)
    # S.correlate_spatial_PVs_by_trial()
    pass
