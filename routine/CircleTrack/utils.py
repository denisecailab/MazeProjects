import os
import re
import shutil
import tkinter as tk
from pathlib import Path
from shutil import copyfile

import cv2
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.stats import circmean, mode
from skimage.feature import register_translation
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB
from tqdm import tqdm

from ..CaImaging.PlaceFields import spatial_bin
from ..CaImaging.util import (concat_avis, find_closest, make_bins,
                              search_for_files, search_for_folders, sync_data)

tkroot = tk.Tk()
tkroot.withdraw()


def circle_sizes(x, y):
    """
    Get the size of the circle track given visited x and y coordinates.
    The mouse must visit all parts of the circle.

    :parameters
    ---
    x, y: array-like
        x, y coordinates

    :return
    (width, height, radius, center): tuple
        Self-explanatory.
    """
    x_extrema = [min(x), max(x)]
    y_extrema = [min(y), max(y)]
    width = np.diff(x_extrema)[0]
    height = np.diff(y_extrema)[0]

    radius = np.mean([width, height]) / 2
    center = [np.mean(x_extrema), np.mean(y_extrema)]

    return (width, height, radius, center)


def batch_concat_avis(mouse_folder):
    """
    Batch concatenates the avi chunks in a mouse folder.

    :parameter
    ---
    mouse_folder: str
        Directory containing session folders. The session folders
        must have the format H??_M??_S??.
    """
    # Recursively search for the session folders.
    folders = [folder for folder in Path(mouse_folder).rglob("H??_M*_S??")]

    # For each folder, check that Merged.avi doesn't already exist.
    for session in folders:
        merged_file = os.path.join(session, "Merged.avi")

        if os.path.exists(merged_file):
            print(f"{merged_file} already exists")

        # If not, concatenate the avis.
        else:
            try:
                concat_avis(
                    session, pattern="behavCam*.avi", fname="Merged.avi", fps=30
                )
            except:
                print(f"Failed to create {merged_file}")


def get_session_folders(mouse_folder: str):
    """
    Find all the session folders within a subtree under mouse_folder.

    :parameter
    ---
    mouse_folder: str
        Folder for a single mouse.

    :return
    ---
    folders: list of Paths
        Directories for each session.
    """
    folders = [folder for folder in Path(mouse_folder).rglob("H??_M*_S??")]

    return folders


def sync(
    minian_folder,
    behavior_data,
    timestamp_path,
    miniscope_cam=5,
    behav_cam=1,
    convert_to_np=True,
):
    """
    Synchronizes minian and behavior files. Does specific correction
    for circle track data. The regular sync_data function downsamples
    the behavior data to match the minian data. However, lick and
    water delivery frames must not be skipped, so we mark the next
    closest frame post-hoc.

    :parameters
    ---
    folder: str
        Path containing all the necessary data files:
            -Behavior csv
            -timestamp.dat
            -minian folder

    csv_fname: str
        File name of the Preprocess() csv.

    timestamp_path: str
        File name of the timestamp file (default from Miniscope
        DAQ is timsetamp.dat).

    miniscope_cam: int
        Camera number corresponding to the miniscope.

    behav_cam: int
        Camera number corresponding to the behavior camera.
    """
    # Sync data by downsampling behavior.
    postsync_behavior, minian, presync_behavior = sync_data(
        behavior_data,
        minian_folder,
        timestamp_path,
        miniscope_cam=miniscope_cam,
        behav_cam=behav_cam,
        convert_to_np=convert_to_np,
    )

    # Find all water delivery frames and relocate them to the
    # next closest frame that survived downsample.
    water_frames = presync_behavior.frame.loc[presync_behavior.water]
    synced_frames = postsync_behavior.frame
    corrected = postsync_behavior.copy()
    for frame in water_frames:
        matching_frame = find_closest(synced_frames, frame, sorted=True)[0]
        corrected.loc[matching_frame, "water"] = True

    # Do the same to lick frames.
    lick_frames = presync_behavior.frame.loc[presync_behavior.lick_port > -1]
    ports = presync_behavior.lick_port.loc[presync_behavior.lick_port > -1]
    for port, frame in zip(ports, lick_frames):
        matching_frame = find_closest(synced_frames, frame, sorted=True)[0]
        corrected.loc[matching_frame, "lick_port"] = port

    return corrected, minian


class SessionStitcher:
    def __init__(
        self,
        folder_list,
        recording_duration,
        miniscope_cam=6,
        behav_cam=2,
        fps=30,
        miniscope_pattern="msCam*.avi",
        behavior_pattern="behavCam*.avi",
    ):
        """
        Combine recording folders that were split as a result of
        the DAQ software crashing. Approach: In a new folder, copy
        all the miniscope files from the first folder, then add a
        file with junk frames that accounts for the time it took
        to reconnect. Then copy all the files from the second folder.
        For the behavior, do the same thing but merge them into one
        file.

        """

        self.folder_list = folder_list
        assert len(folder_list) == 2, "This only works for sessions with 1 crash."

        self.recording_duration = recording_duration
        self.fps = fps
        self.interval = int(np.round((1 / self.fps) * 1000))
        self.camNum = {"miniscope": miniscope_cam, "behavior": behav_cam}
        self.file_patterns = {
            "miniscope": miniscope_pattern,
            "behavior": behavior_pattern,
        }
        self.timestamp_paths = [
            os.path.join(folder, "timestamp.dat") for folder in self.folder_list
        ]

        # Find out how many frames are missing.
        self.missing_frames = self.calculate_missing_frames()
        print(
            f"Estimated missing frames: {int(self.missing_frames)} or "
            f"{np.round(self.missing_frames/30,2)} seconds."
        )
        self.stitched_folder = self.make_stitched_folder()

        print("MERGING TIMESTAMP FILES.")
        self.merge_timestamp_files()
        print("DONE.")

        print("MERGING BEHAVIOR VIDEOS")
        print("If interrupted, delete the last video file in the folder.")
        self.stitch("behavior")
        print("DONE.")

        print("MERGING MINISCOPE VIDEOS")
        print("If interrupted, delete the last video file in the folder")
        self.stitch("miniscope")
        print("DONE.")

        print("CONCATENATING BEHAVIOR VIDEOS")
        print('If interrupted, delete "Merged.avi".')
        concat_avis(self.stitched_folder, pattern=self.file_patterns["behavior"])

        print("ALL PROCESSES COMPLETED.")

    def merge_timestamp_files(self):
        """
        Combine the timestamp.dat files by taking the first file,
        adding in missing data, then time-shifting the second file.

        """
        # Read the first session's timestamp file. Drop the last
        # entry since it's usually truncated.
        session1 = pd.read_csv(self.timestamp_paths[0], sep="\s+")
        session1.drop(session1.tail(1).index, inplace=True)

        missing_data = self.make_missing_timestamps(session1)

        session2 = pd.read_csv(self.timestamp_paths[1], sep="\s+")
        session2 = self.timeshift_second_session(missing_data, session2)

        # Merge.
        df = pd.concat((session1, missing_data, session2))
        path = os.path.join(self.stitched_folder, "timestamp.dat")
        df = df.astype({"frameNum": int, "sysClock": int, "camNum": int})
        df.to_csv(path, sep="\t", index=False)

    def stitch(self, camera):
        """
        Copy session half #1, fill in missing values, copy session
        half #2.

        :parameter
        ---
        camera: str
            'behavior' or 'miniscope'.
        """
        if camera == "behavior":
            self.shifts, self.behav_dims = self.align_projections()

        self.copy_files(self.folder_list[0], camera=camera, second=False)
        self.make_missing_video(camera=camera)
        self.copy_files(self.folder_list[1], camera=camera, second=True)

    def make_missing_timestamps(self, df):
        """
        Build the DataFrame with the missing timestamps.

        :parameter
        ---
        df: DataFrame
            DataFrame from first session.

        :return
        ---
        data: DataFrame
            Missing data buffer.
        """
        # Define the cameras. We can loop through these to shorten the code.
        cameras = ["miniscope", "behavior"]

        # Find the last camera entry from session 1. Convert to str keys
        # used to access dicts.
        last_camera = df.camNum.iloc[-1]
        camera1 = [key for key, value in self.camNum.items() if value != last_camera][0]
        camera2 = [key for key, value in self.camNum.items() if value == last_camera][0]

        # Find the last frames and timestamps from session 1.
        last_frames = {
            camera: df.loc[df.camNum == self.camNum[camera], "frameNum"].iloc[-1] + 1
            for camera in cameras
        }
        last_ts = {
            camera: df.loc[df.camNum == self.camNum[camera], "sysClock"].iloc[-1]
            + self.interval
            for camera in cameras
        }

        # Build new frames and timestamps.
        frames = {
            camera: np.arange(
                last_frames[camera], last_frames[camera] + self.missing_frames
            )
            for camera in cameras
        }
        ts = {
            camera: np.arange(
                last_ts[camera],
                last_ts[camera] + self.interval * self.missing_frames,
                self.interval,
            )
            for camera in cameras
        }

        # Make the DataFrame.
        data = dict()
        data["camNum"] = np.empty((self.missing_frames * 2))
        data["frameNum"] = np.empty_like(data["camNum"])
        data["sysClock"] = np.empty_like(data["camNum"])

        for key, arr in zip(
            ["camNum", "frameNum", "sysClock"], [self.camNum, frames, ts]
        ):
            data[key][::2] = arr[camera1]
            data[key][1::2] = arr[camera2]

        data = pd.DataFrame(data)
        data["buffer"] = 1

        return data

    def timeshift_second_session(self, missing_df, df2):
        """
        Shift the second session in time so that it lines up with the
        missing data.

        :parameters
        ---
        missing_df: DataFrame
            DataFrame from make_missing_timestamps()

        df2: DataFrame
            DataFrame from the second session.

        :return
        ---
        df2: DataFrame
            Timeshifted DataFrame.

        """
        cameras = ["miniscope", "behavior"]

        last_frames = {
            camera: missing_df.loc[
                missing_df.camNum == self.camNum[camera], "frameNum"
            ].iloc[-1]
            for camera in cameras
        }
        last_ts = missing_df.sysClock.iloc[-1]
        df2.sysClock += last_ts

        for camera in cameras:
            df2.loc[df2.camNum == self.camNum[camera], "frameNum"] += last_frames[
                camera
            ]

        # Correct that entry in sysClock that's really high at one of the
        # first two entries of timestamp.dat.
        i = np.argmax(df2.sysClock)
        spike_camera = df2.loc[i, "camNum"]
        spike_ts = df2.loc[df2.camNum == spike_camera, "sysClock"]
        next_ts = spike_ts.iloc[np.argmax(spike_ts) + 1]
        df2.loc[i, "sysClock"] = next_ts - self.interval

        if df2.loc[0, "sysClock"] > df2.loc[1, "sysClock"]:
            df2.loc[0, "sysClock"] = df2.loc[1, "sysClock"] - 1

        return df2

    def calculate_missing_frames(self):
        """
        Calculate the number of missing frames by taking the difference
        between recorded and ideal number of frames.

        :return
        ---
        missing_data_frames: int
            Number of missing frames.
        """
        # First, determine the number of missing frames.
        # Read the timestmap.dat files.
        timestamp_files = [
            os.path.join(folder, "timestamp.dat") for folder in self.folder_list
        ]
        self.last_timestamps = []

        # Get the total number of frames for both session
        # folders.
        for file in timestamp_files:
            df = pd.read_csv(file, sep="\s+")

            self.last_timestamps.append(df.sysClock.iloc[-1])

        # Calculate the discrepancy.
        recorded_duration_s = np.sum(self.last_timestamps) / 1000
        ideal_duration_s = self.recording_duration * 60
        difference = ideal_duration_s - recorded_duration_s
        missing_data_frames = np.round(difference * self.fps).astype(int)

        return missing_data_frames

    def make_stitched_folder(self):
        """
        Makes the folder containing the merged files.

        :return:
        ---
        folder_name: str
        """
        folder_name = self.folder_list[0] + "_stitched"
        try:
            print(f"Creating new directory called {folder_name}.")
            os.mkdir(folder_name)
        except:
            print(f"{folder_name} already exists.")

        return folder_name

    def get_files(self, folder, pattern):
        """
        Find all the files that match the pattern in this folder.

        :parameters
        ---
        folder: str
            Path containing behavior or miniscope videos.

        pattern: regexp str
            For example, 'msCam*.avi'.

        :return
        ---
        files: list
            List of files matching that pattern.
        """
        files = natsorted([str(file) for file in Path(folder).rglob(pattern)])

        return files

    def copy_files(self, source, camera, second=False):
        """
        Copy video files. Renumber if needed.

        :parameters
        ---
        source: str
            Path that you want to copy.

        camera: str
            'miniscope' or 'behavior'

        second: boolean
            Whether you are copying the second part of the session.


        """
        # Find the pattern associated with that video file.
        pattern = self.file_patterns[camera]
        files = self.get_files(source, pattern)
        do_shift = True if second else False

        if camera == "behavior":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            size = self.behav_dims[::-1]

        for file in files:
            fname = os.path.split(file)[-1]

            if second:
                current_number = int(re.findall(r"\d+", fname)[0])
                new_number = current_number + self.last_number
                fname = self.file_patterns[camera].replace("*", str(new_number))

            destination = os.path.join(self.stitched_folder, fname)

            if os.path.isfile(destination):
                print(f"{destination} already exists. Skipping.")
            else:
                if camera == "miniscope":
                    print(f"Copying to {destination}.")
                    copyfile(file, destination)
                elif camera == "behavior":
                    print(f"Copying and cropping to {destination}.")
                    writer = cv2.VideoWriter(destination, fourcc, float(self.fps), size)
                    cap = cv2.VideoCapture(file)
                    cap.set(1, 0)  # First frame.
                    max_frames = int(cap.get(7))

                    for _ in range(max_frames):
                        ret, frame = cap.read()

                        if ret:
                            frame = frame[: self.common_dims[0], : self.common_dims[1]]
                            final_frame = self.correct_frame(frame, do_shift=do_shift)

                            writer.write(final_frame)
                        else:
                            break

                    writer.release()

    def make_missing_video(self, camera):
        """
        Makes the avi file with the missing data (duration of time
        between the crash and getting the data stream started again).
        Currently this writes the last frame before the crash into
        the missing data.

        :parameters
        ---
        camera: str
            'miniscope' or 'behavior'.
        """
        # Find the filename of the last video before the crash.
        # The missing data video will use the next number.
        pattern = self.file_patterns[camera]
        files = self.get_files(self.folder_list[0], pattern)
        last_video = os.path.join(self.stitched_folder, os.path.split(files[-1])[-1])
        last_number = int(re.findall(r"\d+", os.path.split(last_video)[-1])[0])
        self.last_number = last_number + 1

        # Video writing properties.
        cap = cv2.VideoCapture(last_video)
        size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fname = self.file_patterns[camera].replace("*", str(self.last_number))
        full_path = os.path.join(self.stitched_folder, fname)

        if not os.path.exists(full_path):
            print(f"Writing {full_path}.")

            video = cv2.VideoWriter(full_path, fourcc, float(self.fps), size)

            # Read the last frame of the last video.
            cap.set(1, cap.get(7) - 1)
            ret, frame = cap.read()

            for _ in range(self.missing_frames):
                video.write(frame)

            video.release()
        else:
            print(f"{full_path} already exists. Skipping.")

        cap.release()

    def find_smallest_dims(self):
        """
        In order to merge the behavior videos, we must first crop
        and align the two halves of the session. Since the DAQ software
        requires cropping of the behavior camera FOV, it's hard to get
        the exact crop right for both before and after the crash.
        This function finds the dimension sizes that will accommodate
        both recordings. Then we can use those dimensions to crop
        and align the two halves.

        :return
        ---
        dims: [h,w] list
            Smallest dimensions for both height and width for the
            two session halves.

        """
        videos = [
            self.get_files(folder, self.file_patterns["behavior"])[0]
            for folder in self.folder_list
        ]

        # Get the frame size for two videos.
        shapes = []
        for video in videos:
            cap = cv2.VideoCapture(video)
            ret, frame = cap.read()
            shapes.append(frame.shape)

        return np.min(shapes, axis=0)[:2]

    def median_projection(self, video_path, nframes=100, crop=None):
        """
        We will shift each frame in the second half based on a
        cross correlation between their median projections. This
        function calculates the median projection for one video in
        each half.

        :parameters
        ---
        video_path: str
            Path to a video, including file name.

        nframes: int
            Number of frames to subsample to compute median projection.

        crop: (row, col) tuple or None
            Usually will be the smallest dimension sizes that accommodate
            both videos. Get this from self.find_smallest_dims().

        :return
        ---
        proj: (h,w) array
            Median projection.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if crop is not None:
            h, w = crop
        else:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape
        last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        collection = np.zeros((nframes, h, w))
        samples = np.random.randint(0, last_frame, nframes)
        for i, frame in enumerate(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if crop is not None:
                collection[i] = gray[: crop[0], : crop[1]]
            else:
                collection[i] = gray

        cap.release()

        proj = np.median(collection, axis=0)

        return proj

    def align_projections(self, nframes=100):
        """
        Compute the optimal (row, col) translation to align the two
        session halves. Also crops out the strips of the frame that
        get translated.

        :parameter
        ---
        nframes: int
            Number of frames to subsample to compute median projection.

        :return
        ---
        self.shifts: (x, y) tuple
            Optimal shifts.

        new_dims: (h, w) tuple
            New dimensions after cropping away the translated regions.
        """
        self.common_dims = self.find_smallest_dims()
        videos = [
            self.get_files(folder, self.file_patterns["behavior"])[i]
            for folder, i in zip(self.folder_list, [-1, 0])
        ]

        # Get median projections.
        projs = []
        for video in videos:
            projs.append(
                self.median_projection(video, nframes=nframes, crop=self.common_dims)
            )

        # Get optimal shifts.
        shifts = register_translation(projs[0], projs[1], return_error=False)
        self.shifts = shifts.astype(int)

        # Shift and compute new dimension size.
        shifted = self.correct_frame(projs[1], do_shift=True)
        new_dims = shifted.shape

        return self.shifts, new_dims

    def correct_frame(self, img, do_shift=True):
        """
        Crops frame to appropriate size, also aligns the image if
        applicable (session half #2).

        :parameters
        ---
        img: (h, w) array
            Image to align and/or crop.

        do_shift: boolean
            Whether to do the alignment. Only do this for session
            half #2.
        """
        if do_shift:
            shifted = np.roll(img, self.shifts, (0, 1))
        else:
            shifted = img
        final = self.trim(shifted)

        return final

    def trim(self, img):
        """
        Crops the image to the size that remains after alignment
        and cropping.

        :parameter
        ---
        img: (h, w) array
            Image to crop.
        """
        for i, shift in enumerate(self.shifts):
            if shift > 0:
                to_delete = np.arange(0, shift + 1)
            elif shift == 0:
                to_delete = []
            else:
                to_delete = np.arange(-shift, 0)

            img = np.delete(img, to_delete, axis=i)

        return img


class SessionStitcherV4:
    def __init__(self, session_folder):
        self.paths = dict()
        self.paths["date_folder"] = session_folder

        reg_exps = [
            "^H?[0-9]+_M?[0-9]+_S?[0-9]+$",
            "^Miniscope$",
            "^BehavCam_\d{1}$",
        ]

        keys = [
            "session_folders",
            "miniscope",
            "behavior",
        ]

        for key, reg_exp in zip(keys, reg_exps):
            self.paths[key] = search_for_folders(self.paths["date_folder"], reg_exp)

        self.paths["lick_files"] = search_for_files(
            self.paths["date_folder"], "^H\d{2}_M\d{2}_S\d{2}.\d{4} \d{4}.txt$"
        )

        print("Copying video files.")
        self.copy_videos()

        print("Combining timestamps and lick files.")
        self.combine_files()

    def copy_videos(
        self,
        miniscope_fname_pattern="^\d{1,2}.avi$",
        behavior_fname_pattern="^\d{1,2}.avi$",
    ):
        # Get the names of the first folders. We will copy files here.
        self.consolidated_folder_name = {
            data_stream: self.paths[data_stream][0]
            for data_stream in ["miniscope", "behavior"]
        }

        # For both miniscope and behavior folders, copy files.
        for data_stream, pattern in zip(
            ["miniscope", "behavior"], [miniscope_fname_pattern, behavior_fname_pattern]
        ):
            for folder in self.paths[data_stream][1:]:
                # Find the filename of the last file.
                existing_vids = natsorted(
                    search_for_files(
                        self.consolidated_folder_name[data_stream], pattern
                    )
                )
                last_number = self.get_video_number(existing_vids[-1])

                # Get filenames of the videos to copy.
                vids_to_copy = natsorted(search_for_files(folder, pattern))

                # For each video, change the filename to be the next
                # number.
                for vid in vids_to_copy:
                    current_number = self.get_video_number(vid)
                    new_number = current_number + last_number + 1
                    destination = os.path.join(
                        self.consolidated_folder_name[data_stream],
                        str(new_number) + ".avi",
                    )
                    copyfile(vid, destination)
                    print(f"Successfully copied {vid} to {destination}.")

    def get_video_number(self, path):
        # Get the number before .avi.
        n = int(re.findall(r"\d+", os.path.split(path)[-1])[0])

        return n

    def combine_files(self):
        lick_files_concatenated = False
        for data_stream in ["miniscope", "behavior"]:
            # Get the base timestamp file, where we'll be appending to.
            original_fname = os.path.join(self.paths[data_stream][0], "timeStamps.csv")

            original_lick_file = self.paths["lick_files"][0]

            for folder, lick_file in zip(
                self.paths[data_stream][1:], self.paths["lick_files"][1:]
            ):
                # Get the last values from the base file.
                original_df = pd.read_csv(original_fname)
                last_frame = original_df["Frame Number"].iloc[-1]
                last_timestamp = original_df["Time Stamp (ms)"].iloc[-1]

                if not lick_files_concatenated:
                    # Read original lick file.
                    original_lick_df = pd.read_csv(original_lick_file, header=None)
                    original_lick_df.columns = [
                        "Port",
                        "Frame Number",
                        "Time Stamp (ms)",
                    ]

                    # Read lick file to append.
                    lick_df_to_append = pd.read_csv(lick_file, header=None)
                    lick_df_to_append.columns = [
                        "Port",
                        "Frame Number",
                        "Time Stamp (ms)",
                    ]

                    # Add values to frame and timestamps.
                    lick_df_to_append["Frame Number"] += last_frame + 1
                    lick_df_to_append["Time Stamp (ms)"] += last_timestamp + 33

                    # Concatenate.
                    lick_df = pd.concat((original_lick_df, lick_df_to_append))
                    lick_df = lick_df.astype(
                        {
                            "Port": int,
                            "Frame Number": int,
                            "Time Stamp (ms)": int,
                        }
                    )

                    # Save and replace.
                    lick_df.to_csv(original_lick_file, header=False, index=False)
                    print(
                        f"{lick_file} successfully combined with {original_lick_file}"
                    )

                # Get the file to append to.
                append_fname = os.path.join(folder, "timeStamps.csv")
                df_to_append = pd.read_csv(append_fname)

                # Modify the values to add onto the base file.
                df_to_append["Time Stamp (ms)"].iloc[0] = 0
                df_to_append["Time Stamp (ms)"] += last_timestamp + 33
                df_to_append["Frame Number"] += last_frame + 1

                # Concatenate.
                new_df = pd.concat((original_df, df_to_append))
                new_df = new_df.astype(
                    {
                        "Frame Number": int,
                        "Time Stamp (ms)": int,
                    }
                )

                # Save the new timestamp file.
                new_df.to_csv(original_fname, index=False)
                print(f"{append_fname} successfully combined with {original_fname}.")

            lick_files_concatenated = True


def get_equivalent_local_path(folder, local_base=r"D:\Projects", depth=2):
    folder = os.path.normpath(folder)
    split_tree = folder.split(os.sep)

    return os.path.join(local_base, *split_tree[depth:])


def get_circular_error(y_predicted, y_real, n_spatial_bins):
    """
    Error is not linear here, it's circular because for example,
    spatial bin 1 is right next to spatial bin 36. Take the
    circular distance here.

    :parameters
    ---
    y_predicted: array-like of ints
        Predicted spatially binned locations.

    y_real: array-like of ints
        Real spatially binned locations.

    n_spatial_bins: int
        Number of spatial bins (needed for modulo function).
    """
    i = (y_predicted - y_real) % n_spatial_bins
    j = (y_real - y_predicted) % n_spatial_bins

    d = np.min(np.vstack((i, j)), axis=0)

    return d


def format_spatial_location_for_decoder(
        lin_position,
        n_spatial_bins=36,
        time_bin_size=1,
        fps=15,
        classifier=BernoulliNB(),
):
    """
    Naive Bayes classifiers only take integers as outcomes.
    Bin spatial locations both spatially and temporally.

    :parameters
    ---
    lin_position: array-like of floats
        Linearized position (in radians, from behavioral DataFrame).

    n_spatial_bins: int
        Number of spatial bins.

    decoder_time_bin_size: float
        Size of temporal bin in seconds.

    fps: int
        Frames per second of the acquired data.
    """
    # Find the appropriate bin edges given number of spatial bins.
    # Then do spatial bin.
    dont_bin_space = True if isinstance(classifier, (LinearRegression)) else False
    if dont_bin_space:
        binned_position = np.cos(lin_position)
    else:
        bins = np.histogram(lin_position, bins=n_spatial_bins)[1]
        binned_position = np.digitize(lin_position, bins)

    # Do the same for temporal binning.
    bins = make_bins(binned_position, fps * time_bin_size, axis=0)
    binned_position = np.split(binned_position, bins, axis=0)

    # Get the most occupied spatial bin within each temporal bin.
    if dont_bin_space:
        position = np.array([circmean(time_bin) for time_bin in binned_position])
    else:
        position = np.array([mode(time_bin)[0][0] for time_bin in binned_position])

    return position

def find_reward_spatial_bins(lin_position, port_locations, spatial_bin_size_radians=0.05):
    bins = spatial_bin(
        lin_position,
        np.zeros_like(lin_position),
        bin_size_cm=spatial_bin_size_radians,
        show_plot=False,
        one_dim=True,
    )[-1]
    reward_locations_bins = np.digitize(port_locations, bins) - 1

    return reward_locations_bins, bins

def replace_LEDoff_frames(fpath, replacement_frame_number=4):
    folder = os.path.join(os.path.split(fpath)[0], 'originals')
    if not os.path.exists(folder):
        os.mkdir(folder)

    fname = os.path.split(fpath)[1]
    move_fpath = os.path.join(folder, fname)

    # Move the original file.
    if not os.path.exists(move_fpath):
        shutil.move(fpath, move_fpath)
        print(f'Moved {fpath} to {move_fpath}')
    else:
        raise FileExistsError('The folder is already storing an original 0.avi. Aborting to prevent overwrite.')

    compressionCodec = "FFV1"
    codec = cv2.VideoWriter.fourcc(*compressionCodec)

    cap = cv2.VideoCapture(move_fpath)
    rows, cols = int(cap.get(4)), int(cap.get(3))

    # Get replacement frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, replacement_frame_number)
    _, replacement_frame = cap.read()
    replacement_frame = replacement_frame[:,:,1]

    #Return to the first frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writeFile = cv2.VideoWriter(fpath, codec, 60, (cols, rows), isColor=False)
    # Rewrite video file.
    print(f'Writing new {fpath}')
    for frame_number in tqdm(range(int(cap.get(7)))):
        ret, frame = cap.read()

        if ret:
            if frame_number < replacement_frame_number:
                writeFile.write(np.uint8(replacement_frame))
            else:
                writeFile.write(np.uint8(frame[:,:,1]))

        else:
            break

    writeFile.release()
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    dpath = r'Z:\Will\RemoteReversal\Data\Ron\2021_12_13_Goals1\16_04_26\Miniscope'
    replace_LEDoff_frames(os.path.join(dpath,'0.avi'), 4)
    # SessionStitcherV4(folder)


