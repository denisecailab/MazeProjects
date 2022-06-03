import glob
import os
import pickle
from pathlib import Path

import h5py
import hdf5storage
import numpy as np
import pandas as pd
from natsort import natsorted

from Miniscope import open_minian
from plotting import overlay_footprints
from util import ScrollPlot, filter_sessions, nan_array


class SpatialFootprints:
    def __init__(self, mouse_path, session_folder_up_n_levels=-3, minian_perf=True):
        """
        Class that handles spatial footprint-related stuff. Currently only
        makes the .mat file that gets fed into Ziv Lab's CellReg Matlab package.
        ***Currently requires that your files are organized like this***:
        -animal
            -session
                -session_id (which contains minian folder)

        :parameter
        ---
        mouse_path: str, animal folder (see  above)

        """
        # Define paths.
        self.mouse_path = mouse_path
        self.session_paths = [
            folder.parent for folder in Path(self.mouse_path).rglob("minian")
        ]
        self.session_numbers = [
            folder.parts[session_folder_up_n_levels] for folder in self.session_paths
        ]
        self.minian_perf = minian_perf

    def make_mat(self, save_path=None):
        """
        Makes spatial footprints .mat for CellReg Matlab package.

        :parameter
        ---
        save_path: str, path to save .mat file. Defaults to a folder called
            SpatialFootprints inside the session_id folder.
        """
        if save_path is None:
            save_path = os.path.join(self.mouse_path, "SpatialFootprints")

        cellreg_path = os.path.join(save_path, "CellRegResults")

        try:
            os.mkdir(save_path)
        except:
            print("Directory already exists. Proceeding.")

        try:
            os.mkdir(cellreg_path)
        except:
            print("Directory already exists. Proceeding.")

        for session, session_number in zip(self.session_paths, self.session_numbers):
            # File name.
            fname = os.path.join(save_path, session_number + ".mat")

            # Load data.
            data = open_minian(session)

            # Reshape matrix. CellReg reads (neuron, x, y) arrays.
            footprints = np.asarray(data.A)

            if not self.minian_perf:
                footprints = np.rollaxis(footprints, 2)

            # Save.
            matfiledata = {}
            matfiledata[u'footprints'] = footprints
            hdf5storage.writes(mdict=matfiledata, filename=Path(fname))
            print(f"Saved {fname}")


class CellRegObj:
    def __init__(self, path):
        """
        Object for handling and saving outputs from CellReg Matlab package.

        :parameter
        ---
        path: str, full path to CellRegResults folder.

        """
        self.path = path
        self.sessions = self.get_sessions()
        try:
            self.map = load_cellreg_results(self.path)
        except:
            self.data, self.file = self.read_cellreg_output()
            self.compile_cellreg_data()
            self.map = load_cellreg_results(self.path)

    def get_sessions(self):
        """
        Get sessions by going up a level on the path and searching for
        mat files.

        :return
        ---
        sessions: list
            Session names.
        """
        mat_files = glob.glob(os.path.join(os.path.split(self.path)[0], "*.mat"))

        # Based on comments from Y.Z. and D.M., CellReg sorts the sessions, so
        # we must also sort.
        sessions = natsorted([os.path.split(os.path.splitext(i)[0])[-1] for i in mat_files])

        return sessions

    def read_cellreg_output(self):
        """
        Reads the .mat file.
        :return:
        """
        cellreg_file = glob.glob(os.path.join(self.path, "cellRegistered*.mat"))
        assert len(cellreg_file) > 0, "No registration .mat detected."
        assert len(cellreg_file) is 1, "Multiple cell registration files!"
        cellreg_file = cellreg_file[0]

        # Load it.
        file = h5py.File(cellreg_file)
        data = file["cell_registered_struct"]

        return data, file

    def process_registration_map(self):
        # Get the cell_to_index_map. Reading the file transposes the
        # matrix. Transpose it back.
        cell_to_index_map = self.data["cell_to_index_map"].value.T

        # Matlab indexes starting from 1. Correct this for Python.
        # Then NaN out the unmatched cells.
        match_map = cell_to_index_map - 1
        match_map[match_map == -1] = -9999

        # Convert to DataFrame and make the column names the sessions.
        assert match_map.shape[1] == len(self.sessions), "Sessions don't match."
        match_map = pd.DataFrame(match_map)
        match_map.columns = self.sessions

        return match_map.astype("int")

    def process_spatial_footprints(self):
        # Get the spatial footprints after translations.
        footprints_reference = self.data["spatial_footprints_corrected"].value[0]

        footprints = []
        for idx in footprints_reference:
            # Float 32 takes less memory.
            session_footprints = np.float32(
                np.transpose(self.file[idx].value, (2, 1, 0))
            )
            footprints.append(session_footprints)

        return footprints

    def process_centroids(self):
        # Also get centroid positions after translations.
        centroids_reference = self.data["centroid_locations_corrected"].value[0]

        centroids = []
        for idx in centroids_reference:
            session_centroids = self.file[idx].value.T
            centroids.append(session_centroids)

        return centroids

    def compile_cellreg_data(self):
        # Gets registration information. So far, this consists of the
        # cell to index cell_map, centroids, and spatial footprints.
        match_map = self.process_registration_map()
        centroids = self.process_centroids()
        footprints = self.process_spatial_footprints()

        filename = os.path.join(self.path, "CellRegResults.csv")
        filename_footprints = os.path.join(self.path, "CellRegFootprints.pkl")
        filename_centroids = os.path.join(self.path, "CellRegCentroids.pkl")

        match_map.to_csv(filename, index=False)
        with open(filename_footprints, "wb") as output:
            pickle.dump(footprints, output, protocol=4)
        with open(filename_centroids, "wb") as output:
            pickle.dump(centroids, output, protocol=4)


def load_cellreg_results(path, mode="cell_map"):
    """
    After having already running CellRegObj, load the saved pkl file.

    """
    # Get file name based on mode.
    file_dict = {
        "cell_map": "CellRegResults.csv",
        "footprints": "CellRegFootprints.pkl",
        "centroids": "CellRegCentroids.pkl",
    }
    fname = os.path.join(path, file_dict[mode])

    # Open pkl file.
    if mode in ["footprints", "centroids"]:
        with open(fname, "rb") as file:
            data = pickle.load(file)
    elif mode == "cell_map":
        data = pd.read_csv(fname)
    else:
        raise KeyError(
            f"{mode} not supported. Use cell_map, footprints, " f"or centroids."
        )

    return data


def get_cellmap_columns(cell_map, cols):
    """
    Gets the exact column name of the cell_map DataFrame given a
    partial str.

    :parameters
    ---
    cell_map: (neuron, session) DataFrame
        Neuron mappings from CellRegObj.

    cols: list or list-like of strs
        Strings corresponding to the session names you want.

    :return
    ---
    sessions: list of strs
        Strings of the actual session names.

    """
    if type(cols) is not list:
        cols = [cols]

    cols = [c.lower() for c in cols]

    sessions = []
    for col in cols:
        sessions.extend([c for c in cell_map.columns if col in c.lower()])

    return sessions


def trim_map(cell_map, cols, detected="everyday"):
    """
    Eliminates columns in the neuron mapping array.

    :parameters
    ---
    cell_map: (neuron, session) DataFrame
        Neuron mappings.

    cols: list-like
        Columns of cell_map to keep.

    detected: str
        'everyday': keeps only the neurons that were detected on
        each session (column) specified.
        'either_day': keeps the neurons that were detected on either
        session specified.

    :return
    ---
    trimmed_map: (neuron, session) array
        Neuron mappings where number of columns is equal to length
        of cols.

    """
    # Take only specified columns. Need to be able to rearrange sessions to not be in sorted order!
    sessions = get_cellmap_columns(cell_map, cols)
    trimmed_map = cell_map[sessions]

    # Eliminate neurons that were not detected on every session.
    if detected == "everyday":
        neurons_detected = (trimmed_map != -9999).all(axis=1)
    elif detected == "either_day":
        neurons_detected = (trimmed_map != -9999).any(axis=1)
    elif detected == "first_day":
        neurons_detected = trimmed_map[trimmed_map.columns[0]] != -9999
    else:
        raise TypeError("Invalid value for detected")

    trimmed_map = trimmed_map.loc[neurons_detected, :]

    return trimmed_map


def rearrange_neurons(cell_map, neural_data):
    """
    Rearranges (neuron, time) arrays by rows to match the mapping.

    :parameters
    ---
    cell_map: (neuron, session) array
        Neuron mappings.

    neural_data: list of (neuron, time) arrays
        Neural data (e.g., S). List must be in same order as the
        columns in cell_map.

    """
    # Handles cases where only one session was fed in.
    cell_map = (
        np.asarray(cell_map, dtype=int)
        if type(cell_map) in [pd.DataFrame, pd.Series]
        else cell_map
    )
    if cell_map.ndim == 1:
        cell_map = np.expand_dims(cell_map, 1)
    neural_data = [neural_data] if not isinstance(neural_data, list) else neural_data

    spike_list = True if isinstance(neural_data[0], list) else False

    missing = -9999
    if not missing in cell_map:
        # Do a simple rearrangement of rows based on mappings.
        rearranged = []
        if spike_list:
            for n, single_session_activity in enumerate(neural_data):
                rearranged.append([single_session_activity[neuron] for neuron in cell_map[:, n]])
        else:
            for n, single_session_activity in enumerate(neural_data):
                rearranged.append(single_session_activity[cell_map[:, n]])

    else:
        # Catches cases where -9999s (non-matched neurons) are in cell_map.
        # In this case, iterate through each cell_map column (session)
        # and then iterate through each neuron. Grab it from neural_data
        # if it exists, otherwise, fill it with zeros.
        print("Unmatched neurons detected in cell_map. Padding with zeros")
        rearranged = []

        if spike_list:
            for n, single_session_activity in enumerate(neural_data):
                for neuron in cell_map[:, n]:
                    if neuron == missing:
                        rearranged.append([])
                    else:
                        rearranged.append(single_session_activity[neuron])
        else:
            for n, single_session_activity in enumerate(neural_data):
                # Assume neuron is not active, fill with zeros.
                rearranged_session = nan_array((cell_map.shape[0], single_session_activity.shape[1]))

                # Be sure to only grab neurons if cell_map value is > -1.
                # Otherwise, it will take the last neuron (-1).
                for m, neuron in enumerate(cell_map[:, n]):
                    if neuron > missing:
                        rearranged_session[m] = single_session_activity[neuron]

                # Append rearranged matrix with paddded zeros.
                rearranged.append(rearranged_session)

    return rearranged

# DEPRECATED.
def get_cellreg_path(cell_map, mouse, animal_key="Mouse", cellreg_key="CellRegPath"):
    """
    Grabs the path containing CellRegResults folder from a dict
    made by dir_dict().

    :parameters
    ---
    mouse: str
        Mouse name.

    dict_list: list of dicts
        From dir_dict(). Should contain a key 'Animal' or something
        similar denoting animal name.

    animal_key: str
        A dict key labeling your animals.

    cellreg_key: str
        A dict key labeling the path to CellRegResults.

    :return
    ---
    path: str
        Path to CellRegResults folder.

    """
    entries = filter_sessions(cell_map, key=animal_key, keywords=mouse)
    path = entries[cellreg_key].iloc[0]

    return path


def scrollplot_footprints(cellreg_path, sessions, neurons=range(10)):
    # Turn a single neuron into a list.
    if isinstance(neurons, int):
        neurons = [neurons]

    # Load map and trim.
    cellreg_map = load_cellreg_results(cellreg_path, mode="cell_map")
    trimmed_map = trim_map(cellreg_map, sessions, detected="everyday")

    # Load footprints.
    footprints = load_cellreg_results(cellreg_path, mode="footprints")
    footprints = [
        footprints_this_day
        for footprints_this_day, session in zip(footprints, cellreg_map)
        if session in sessions
    ]

    mapped_footprints = [
        footprints_this_day[trimmed_map[session]]
        for session, footprints_this_day in zip(trimmed_map, footprints)
    ]

    ScrollObj = ScrollPlot(
        overlay_footprints, footprints=mapped_footprints, figsize=(14, 10)
    )

    return ScrollObj


if __name__ == "__main__":
    # cellreg_path = (
    #     r"Z:\Will\Drift\Data\Encedalus_Scope14\SpatialFootprints\CellRegResults"
    # )

    # CellRegObj(r'Z:\Will\Drift\Data\Encedalus_Scope14\SpatialFootprints\CellRegResults')
    S = SpatialFootprints(r'Z:\Will\Drift\Data\Castor_Scope05')
    S.session_paths = S.session_paths[2:]
    # S.session_numbers = [folder.parts[-3] for folder in
    #                      S.session_paths]
    # S.make_mat()
