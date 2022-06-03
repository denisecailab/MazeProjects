import glob
import os
import tkinter as tk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter

from util import ScrollPlot, consecutive_dist, disp_frame

tkroot = tk.Tk()
tkroot.withdraw()


def read_eztrack(csv_fname):
    """
    Reads ezTrack outputs.

    Parameters
    ---
    csv_fname: str, path to tracking .csv from ezTrack.
    cm_per_pixel: float, centimeters per pixel.

    Return
    ---
    position: dict, with keys x, y, frame, distance.
    """
    # Open file.
    df = pd.read_csv(csv_fname)

    # Consolidate into dict.
    position = {
        "x": np.asarray(df["X"]),  # x position
        "y": np.asarray(df["Y"]),  # y position
        "frame": np.asarray(df["Frame"]),  # Frame number
        "distance": np.asarray(df["Distance_px"]),
    }  # Distance traveled since last sample

    return pd.DataFrame(position)


def make_tracking_video(
    vid_path, csv_path, output_fname="Tracking.avi", start=0, stop=None, fps=30
):
    """
    Makes a video to visualize licking at water ports and position of the animal.

    :parameters
    ---
    video_fname: str
        Full path to the behavior video.

    csv_fname: str
        Full path to the csv file from EZTrack.

    output_fname: str
        Desired file name for output. It will be saved to the same folder
        as the data.

    start: int
        Frame to start on.

    stop: int or None
        Frame to stop on or if None, the end of the movie.

    fps: int
        Sampling rate of the behavior camera.
    """
    # Get behavior video.
    vid = cv2.VideoCapture(vid_path)
    if stop is None:
        stop = int(vid.get(7))  # 7 is the index for total frames.

    # Save data to the same folder.
    folder = os.path.split(vid_path)[0]
    output_path = os.path.join(folder, output_fname)

    # Get EZtrack data.
    eztrack = read_eztrack(csv_path)

    # Make video.
    fig, ax = plt.subplots()
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, output_path, 100):
        for frame_number in np.arange(start, stop):
            # Plot frame.
            vid.set(1, frame_number)
            ret, frame = vid.read()
            ax.imshow(frame)

            # Plot position.
            x = eztrack.at[frame_number, "x"]
            y = eztrack.at[frame_number, "y"]
            ax.scatter(x, y, marker="+", s=60, c="r")

            ax.text(
                0,
                0,
                "Frame: "
                + str(frame_number)
                + "   Time: "
                + str(np.round(frame_number / 30, 1))
                + " s",
            )

            ax.set_aspect("equal")
            plt.axis("off")

            writer.grab_frame()

            plt.cla()


def spatial_bin(
    x,
    y,
    bin_size_cm=20,
    show_plot=False,
    weights=None,
    ax=None,
    bins=None,
    one_dim=False,
    nbins=None,
):
    """
    Spatially bins the position data.

    :parameters
    ---
    x,y: array-like
        Vector of x and y positions in cm.

    bin_size_cm: float
        Bin size in centimeters.

    show_plot: bool
        Flag for plotting.

    weights: array-like
        Vector the same size as x and y, describing weights for
        spatial binning. Used for making place fields (weights
        are booleans indicating timestamps of activity).

    ax: Axis object
        If you want to reference an exis that already exists. If None,
        makes a new Axis object.

    :returns
    ---
    H: (nx,ny) array
        2d histogram of position.

    xedges, yedges: (n+1,) array
        Bin edges along each dimension.
    """
    # Calculate the min and max of position.
    x_extrema = [min(x), max(x)]
    y_extrema = [min(y), max(y)]

    if nbins is None:
        nbins = int(np.round(np.diff(x_extrema)[0] / bin_size_cm))

    if one_dim:
        if bins is None:
            bins = np.linspace(x_extrema[0], x_extrema[1], nbins)

        H, edges = np.histogram(x, bins, weights=weights)

        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()

            ax.plot(H)

        return H, edges, bins

    else:
        if bins is None:
            # Make bins.
            xbins = np.linspace(x_extrema[0], x_extrema[1], nbins)
            ybins = np.linspace(y_extrema[0], y_extrema[1], nbins)
            bins = [ybins, xbins]

        # Do the binning.
        H, xedges, yedges = np.histogram2d(y, x, bins, weights=weights)

        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()

            ax.imshow(H)

        return H, xedges, yedges, bins


class ManualCorrect:
    def __init__(self, video_dict, start_frame=0):
        """
        Manually correct errors in ezTrack.

        :parameters
        ---
        video_dict: dict
            Contains metadata about the video file. Follow the format from
            the ipynb.

        start_frame: int
            Frame where you want to start correction.
        """
        self.video_dict = video_dict
        self.start_frame = start_frame
        self.frame_num = self.start_frame
        # Get csv file path.
        try:
            self.video_dict["csv_fpath"] = os.path.splitext(video_dict["fpath"])
        except:
            video_dict["fpath"] = os.path.join(
                os.path.normpath(video_dict["dpath"]), video_dict["file"]
            )
            self.video_dict["csv_fpath"] = glob.glob(
                os.path.join(video_dict["dpath"], "*LocationOutput.csv")
            )[0]

        # Read csv.
        try:
            self.df = pd.read_csv(self.video_dict["csv_fpath"])
        except FileNotFoundError as not_found:
            print(not_found.filename + " not found. Run ezTrack first")

    def plot(self, frame_num):
        """
        Plot frame and position from ezTrack csv.

        :parameter
        frame_num: int
            Frame number that you want to start on.
        """
        vid = cv2.VideoCapture(self.video_dict["fpath"])
        n_frames = int(vid.get(7))
        frame_nums = ["Frame " + str(n) for n in range(n_frames)]
        self.f = ScrollPlot(
            disp_frame,
            current_position=frame_num,
            vid_fpath=self.video_dict["fpath"],
            x=self.df["X"],
            y=self.df["Y"],
            titles=frame_nums,
        )

    def correct_position(self, start_frame=None):
        """
        Correct position starting from start_frame. If left to default,
        start from where you specified during class instantiation or where
        you last left off.

        :parameter
        ---
        start_frame: int
            Frame number that you want to start on.
        """
        # Frame to start on.
        if start_frame is None:
            start_frame = self.frame_num

        # Plot frame and position, then connect to mouse.
        self.plot(start_frame)
        self.f.fig.canvas.mpl_connect("button_press_event", self.correct)

        # Wait for click.
        while plt.get_fignums():
            plt.waitforbuttonpress()

        # When done, hit Esc on keyboard and csv will be overwritten.
        print("Saving to " + self.video_dict["csv_fpath"])
        self.df.to_csv(self.video_dict["csv_fpath"])

    def correct(self, event):
        """
        Defines what happens during mouse clicks.

        :parameter
        ---
        event: click event
            Defined by mpl_connect. Don't modify.
        """
        # Overwrite DataFrame with new x and y values.
        self.df.loc[self.f.current_position, "X"] = event.xdata
        self.df.loc[self.f.current_position, "Y"] = event.ydata

        # Plot the new x and y.
        self.f.fig.axes[0].plot(event.xdata, event.ydata, "go")
        self.f.fig.canvas.draw()

    def find_jerks(self):
        """
        Plot the distances between points and then select with your
        mouse where you would want to do a manual correction from.

        """
        # Plot distance between points and connect to mouse.
        # Clicking the plot will bring you to the frame you want to
        # correct.
        self.velocity_fig, self.velocity_ax = plt.subplots(1, 1)
        try:
            self.velocity_ax.plot(self.df["Distance_in"])
        except:
            self.velocity_ax.plot(self.df["Distance_px"])
        self.velocity_fig.canvas.mpl_connect("button_press_event", self.jump_to)

        while plt.get_fignums():
            plt.waitforbuttonpress()

    def jump_to(self, event):
        """
        Jump to this frame based on a click on a graph. Grabs the x (frame)
        """
        plt.close(self.velocity_fig)
        self.correct_position(int(np.round(event.xdata)))


def convert_dlc_to_eztrack(h5_path: str, save_path=None):
    """
    Converts DeepLabCut outputs into a format similar to ezTrack's.
    This is motivated by most of my existing code already
    accommodating ezTrack files.

    :parameter
    ---
    h5_path: str
        Path to the h5 file containing the output of DLC.

    :return
    ---
    data: DataFrame
        -x, y: Coordinates of body.
        -distance: Distance traveled since last sample.
        -frame: Frame number.
        -*body_part*_x, *body_part*_y: If any other body
            parts were labeled by DLC, their coordinates
            will also be the body part name + _x, or _y.
        -*body_part*_distance: Same for distance.
    """
    # Read the h5 file and get the keys.
    df = pd.read_hdf(h5_path)
    scorer = df.columns.levels[0][0]
    body_parts = df.columns.levels[1]

    # If 'body' was not specified, assume the first body
    # part is the body.
    if "body" not in body_parts:
        print("Warning! body is not one of the labeled body parts")
        print("Proceeding as if the first labeled body part is the body.")
        body = body_parts[0]
    else:
        body = "body"

    # Extract the data. Start with body.
    x = df[scorer][body]["x"]  # x coordinate
    y = df[scorer][body]["y"]  # y coordinate
    frames = [i for i in range(len(df[scorer][body]["x"]))]  # frame number
    distances = consecutive_dist(np.asarray((x, y)).T)  # distance from last sample
    distances = np.insert(distances, 0, 0)  # First distance is 0.
    new_df = {"X": x, "Y": y, "Frame": frames, "Distance_px": distances}

    # Then do the other body parts.
    for part in body_parts:
        if part != "body":
            x = df[scorer][part]["x"]
            y = df[scorer][part]["y"]
            distances = consecutive_dist(np.asarray((x, y)).T)
            distances = np.insert(distances, 0, 0)

            new_df[part + "_X"] = x
            new_df[part + "_Y"] = y
            new_df[part + "_Distance"] = distances

    # Turn into Dataframe.
    data = pd.DataFrame(new_df)

    # Save the csv.
    if save_path is None:
        base_path = os.path.splitext(h5_path)[0]
        save_path = base_path + "_LocationOutput.csv"

    data.to_csv(save_path)

    return data, save_path


if __name__ == "__main__":
    folder = r"D:\Projects\CircleTrack\Mouse4\01_29_2020\H15_M20_S48"
    fname = "MergedDLC_resnet50_circletrackFeb4shuffle1_218500.h5"
    path = os.path.join(folder, fname)
    convert_dlc_to_eztrack(path)
