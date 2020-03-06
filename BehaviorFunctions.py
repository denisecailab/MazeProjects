import os
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from LickArduino import clean_Arduino_output
from util import read_eztrack, find_closest, ScrollPlot, disp_frame, \
    consecutive_dist, sync_cameras, nan_array
from scipy.stats import zscore
from scipy.stats import norm
import matplotlib.pyplot as plt
import cv2
from CircleTrack.utils import circle_sizes, cart2pol, grab_paths, convert_dlc_to_eztrack
import tkinter as tk
tkroot = tk.Tk()
tkroot.withdraw()
from tkinter import filedialog
from scipy.ndimage import gaussian_filter1d

def make_tracking_video(vid_path, preprocessed=True, csv_path=None,
                        Arduino_path=None, output_fname='Tracking.avi',
                        start=0, stop=None, fps=30):
    """
    Makes a video to visualize licking at water ports and position of the animal.

    :parameters
    ---
    video_fname: str
        Full path to the behavior video.

    csv_fname: str
        Full path to the csv file containing position (x and y).

    output_fname: str
        Desired file name for output. It will be saved to the same folder
        as the data.

    start: int
        Frame to start on.

    stop: int or None
        Frame to stop on or if None, the end of the movie.

    fps: int
        Sampling rate of the behavior camera.

    Arduino_path:
        Full path to the Arduino output txt. If None, doesn't plot licking.

    """
    # Get behavior video.
    vid = cv2.VideoCapture(vid_path)
    if stop is None:
        stop = int(vid.get(7))  # 7 is the index for total frames.

    # Save data to the same folder.
    folder = os.path.split(vid_path)[0]
    output_path = os.path.join(folder, output_fname)

    # Get EZtrack data.
    if preprocessed:
        session_folder = os.path.split(vid_path)[0]
        behav = Preprocess(session_folder)
        eztrack = behav.behavior_df
    else:
        if Arduino_path is not None:
            eztrack = sync_Arduino_outputs(Arduino_path, csv_path)[0]
            eztrack = clean_lick_detection(eztrack)
        else:
            eztrack = read_eztrack(csv_path)

    # Define the colors that the cursor will flash for licking each port.
    port_colors = ['saddlebrown',
                   'red',
                   'orange',
                   'yellow',
                   'green',
                   'blue',
                   'darkviolet',
                   'gray']

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
            x = eztrack.at[frame_number, 'x']
            y = eztrack.at[frame_number, 'y']
            ax.scatter(x, y, marker='+', s=60, c='w')

            ax.text(0, 0, 'Frame: ' + str(frame_number) +
                    '   Time: ' + str(np.round(frame_number/30, 1)) + ' s')

            # Lick indicator.
            if (Arduino_path is not None) or preprocessed:
                licking_at_port = eztrack.at[frame_number, 'lick_port']
                if licking_at_port >= 0:
                    ax.scatter(x, y, s=200, marker='+',
                               c=port_colors[licking_at_port])

            ax.set_aspect('equal')
            ax.set_ylim([frame.shape[0],0])
            ax.set_xlim([0,frame.shape[1]])
            plt.axis('off')

            writer.grab_frame()

            plt.cla()


def sync_Arduino_outputs(Arduino_fpath, behavior_fpath, behav_cam=2,
                         miniscope_cam=6, sync_mode='timestamp'):
    """
    This function is meant to be used in conjunction with the above
    functions and Miniscope software recordings. Miniscope software
    will save videos (behavior and imaging) in a timestamped folder.

    :param folder:
    :return:
    """
    # Read the txt file generated by Arduino.
    Arduino_data, offset = clean_Arduino_output(Arduino_fpath)
    Arduino_data['Timestamp'] -= offset

    # Read the position data generated by ezTrack.
    behavior_df = read_eztrack(behavior_fpath)
    behavior_df['water'] = False
    behavior_df['lick_port'] = int(-1)

    # Get timestamping information from DAQ output.
    folder = os.path.split(behavior_fpath)[0]
    timestamp_fpath = os.path.join(folder, 'timestamp.dat')
    try:
        sync_map, DAQ_data = sync_cameras(timestamp_fpath,
                                          miniscope_cam=miniscope_cam,
                                          behav_cam=behav_cam)
        sync_map = np.asarray(sync_map)
    except:
        raise Exception('DAQ timestamp.dat not found or corrupted.')

    # Only take the rows corresponding to the behavior camera.
    DAQ_data = DAQ_data[DAQ_data.camNum == behav_cam]
    DAQ_data.reset_index(drop=True, inplace=True)
    DAQ_data.astype({'frameNum': int,
                     'sysClock': int,
                     'camNum': int})

    # Discard data after Miniscope acquisition has stopped.
    sysClock = np.asarray(DAQ_data.sysClock)
    Arduino_data.drop(Arduino_data.index[Arduino_data.Timestamp >
                                         DAQ_data.sysClock.iloc[-1]],
                      inplace=True)

    if sync_mode == 'frame':
        if behav_cam > miniscope_cam:
            miniscope_col, behav_col = 0, 1
        else:
            miniscope_col, behav_col = 1, 0
    else:
        miniscope_col, behav_col = None, None

    # Find the frame number associated with the timestamp of a lick.
    nframes = len(behavior_df)
    print(f'Syncing {Arduino_fpath} to {behavior_fpath} using {sync_mode}s...')
    for i, row in Arduino_data.iterrows():
        if sync_mode == 'timestamp':
            closest_time = find_closest(sysClock, row.Timestamp, sorted=True)[0]
            behav_frame = DAQ_data.loc[closest_time]['frameNum']
        elif sync_mode == 'frame':
            behav_frame = sync_map[sync_map[:,miniscope_col]==row.Frame, behav_col]
        else:
            raise ValueError('sync_mode must be "timestamp" or "frame"')

        if not isinstance(behav_frame, (int, float)):
            behav_frame = behav_frame[0]

        if behav_frame >= nframes:
            continue

        val = row.Data
        behavior_df.at[behav_frame, 'lick_port'] = val
        if val == -1 or val == 'Water':
            behavior_df.at[behav_frame, 'water'] = True

    behavior_df.loc[behavior_df['lick_port'] == 'Water', 'lick_port'] = -1
    behavior_df = behavior_df.astype({'frame': int,
                                      'water': bool,
                                      'lick_port': int})
    return behavior_df, Arduino_data


def find_water_ports(behavior_df, use_licks=True):
    """
    Use the x and y extrema to locate water port locations. Requires that the
    maze be positioned so that a port is at the 12 o'clock position. Which port
    is not important -- the code can be modified for any orientation.

    :parameter
    ---
    behavior_df: cleaned DataFrame from sync_Arduino_outputs()

    :return
    ---
    ports: DataFrame
        DataFrame with 'x' and 'y' columns corresponding to x and y positions of
        each water port.
    """
    (width, height, radius, center) = circle_sizes(behavior_df.x, behavior_df.y)
    theta = np.pi/4     # Angle in between each water port.

    # Determines orientation of the water ports.
    # List the port number of the port at 12 o'clock and count up.
    orientation = [7, 0, 1, 2, 3, 4, 5, 6]
    port_angles = [o * theta for o in orientation]

    # Calculate port locations.
    ports = {}
    ports['x'] = radius * np.cos(port_angles) + center[0]
    ports['y'] = radius * np.sin(port_angles) + center[1]
    ports = pd.DataFrame(ports)

    # So actually the above was an overly complicated way to find the ports
    # and critically depends on the ports being aligned to the appropriate
    # locations on the camera's FOV. A more reliable way would be to just
    # find where the mouse licked. Default to the above if mouse doesn't lick
    # at particular ports.
    if use_licks:
        for port in range(8):
            try:
                licking = behavior_df['lick_port'] == port
                x = np.median(behavior_df.loc[licking, 'x'])
                y = np.median(behavior_df.loc[licking, 'y'])

                ports.loc[port, 'x'] = x
                ports.loc[port, 'y'] = y

                port_angles[port] = linearize_trajectory(behavior_df, x, y)[0]
            except:
                print(f'Port {port} not detected. Using default location.')


    # Debugging purposes.
    # port_colors = ['saddlebrown',
    #                'red',
    #                'orange',
    #                'yellow',
    #                'green',
    #                'blue',
    #                'darkviolet',
    #                'gray']
    # plt.plot(behavior_df.x, behavior_df.y)
    # plt.scatter(center[0], center[1], c='r')
    # for color, (i, port) in zip(port_colors, ports.iterrows()):
    #     plt.scatter(port['x'], port['y'], c=color)
    # plt.axis('equal')

    return ports, port_angles


def clean_lick_detection(behavior_df, threshold=80):
    """
    Clean lick detection data by checking that the mouse is near the port during
    a detected lick.

    :parameters
    ---
    behavior_df: cleaned DataFrame from sync_Arduino_outputs()

    threshold: float
        Distance threshold (in pixels) to be considered "near" the port.

    :return
    ---
    behavior_df: cleaned DataFrame after eliminating false positives.
    """
    ports = find_water_ports(behavior_df)[0]

    lick_frames = behavior_df[behavior_df.lick_port > -1]
    for i, frame in lick_frames.iterrows():
        frame = frame.copy()
        port_num =  frame.lick_port
        frame_num = frame.frame

        distance = np.sqrt((frame.x - ports.at[port_num, 'x'])**2 +
                           (frame.y - ports.at[port_num, 'y'])**2)

        if distance > threshold:
            behavior_df.at[frame_num, 'lick_port'] = -1

    return behavior_df


def linearize_trajectory(behavior_df, x=None, y=None):
    """
    Linearizes circular track trajectory.

    :parameter
    ---
    behavior_df: output from read_eztrack()

    :returns
    ---
    angles: array
        Basically the linearized trajectory. Technically it is the
        polar coordinate with the center of the maze as the origin.

    radii: array
        Vector length of polar coordinate. Basically the distance from
        the center. Maybe useful for something.
    """
    # Get circle size.
    if x is None:
        x = behavior_df.x
    if y is None:
        y = behavior_df.y
    (width, height, radius, center) = circle_sizes(behavior_df.x,
                                                   behavior_df.y)

    # Convert to polar coordinates.
    angles, radii = cart2pol(x-center[0], y-center[1])

    # Shift everything so that 12 o'clock (pi/2) is 0.
    angles += np.pi/2
    angles = np.mod(angles, 2*np.pi)

    return angles, radii


def plot_licks(behavior_df):
    """
    Plot points where mouse licks.

    :parameter
    ---
    behavior_df: output from Preprocess

    :return
    ---
    fig, ax: Figure and Axes
        Contains the plots.
    """
    # Make sure licks have been retrieved.
    if 'lick_port' not in behavior_df:
        raise KeyError('Run sync_Arduino_outputs and clean_lick_detection first.')
    else:
        licks = behavior_df.lick_port
        licks[licks == -1] = np.nan

    # Linearize mouse's trajectory.
    lin_dist = linearize_trajectory(behavior_df)[0]

    # Find the water ports and get their linearized location.
    ports, lin_ports = find_water_ports(behavior_df)

    # Make the array for plotting.
    licks = [lin_ports[port_id] if not np.isnan(port_id) else np.nan for port_id in licks]

    # Plot.
    fig, ax = plt.subplots()
    ax.plot(lin_dist)
    ax.plot(licks, marker='x', markersize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Linearized distance (radians)')

    return fig, ax


def find_rewarded_ports(behavior_df):
    """
    Find which port numbers are rewarded by looking at the flag
    one timestamp before water delivery. Note that the mouse must
    lick at each rewarded port a handful of times for this to work.

    :parameter
    ---
    behavior_df: output from Preprocess()

    :return
    ---
    ports: array
        Port numbers that were rewarded.
    """
    if 'water' not in behavior_df:
        raise KeyError('Run sync_Arduino_outputs and clean_lick_detection first.')

    # Get index one before water delivery (the lick that triggered it).
    one_before = np.where(behavior_df.water)[0]

    # Find unique port numbers.
    rewarded_ports = np.unique(behavior_df.loc[one_before, 'lick_port'])

    return rewarded_ports[rewarded_ports > -1]


def bin_position(linearized_position):
    """
    Bin radial position.

    :parameter
    ---
    linearized_position: array
        Linearized position (position in radians after passing through linearize_trajectory())

    :return
    ---
    binned: array
        Binned position.
    """
    bins = np.linspace(0, 2*np.pi, 9)
    binned = np.digitize(linearized_position, bins)

    return binned


def get_trials(behavior_df, counterclockwise=False):
    """
    Labels timestamps with trial numbers. Looks through position indices as the mouse
    passes through bins in a clockwise fashion (default).

    :parameters
    ---
    behavior_df: output from Preprocess()

    counterclockwise: boolean
        Flag for whether session was run with the mouse running counterclockwise.

    :return
    trials: array, same size as behavior_df position
        Labels for each timestamp for which trial the mouse is on.
    """
    # Linearize then bin position into one of 8 bins.
    position = linearize_trajectory(behavior_df)[0]
    binned_position = bin_position(position)
    bins = np.unique(binned_position)

    # For each bin number, get timestamps when the mouse was in that bin.
    indices = [np.where(binned_position == this_bin)[0] for this_bin in bins]
    if counterclockwise: # reverse the order of the bins.
        indices = indices[::-1]

    # Preallocate trial vector.
    trials = np.full(binned_position.shape, np.nan)
    trial_start = 0
    last_idx = 0

    # We need to loop through bins rather than simply looking for border crossings
    # because a mouse can backtrack, which we wouldn't want to count.
    # For a large number of trials...
    for trial_number in range(500):

        # For each bin...
        for this_bin in indices:
            # Find the first timestamp that comes after the first timestamp in the last bin
            # for that trial. Argmax is supposedly faster than np.where.
            last_idx = this_bin[np.argmax(this_bin > last_idx)]

        # After looping through all the bins, remember the last timestamp where there
        # was a bin transition.
        trial_end = last_idx

        # If the slice still has all NaNs, label it with the trial number.
        if np.all(np.isnan(trials[trial_start:trial_end])):
            trials[trial_start:trial_end] = trial_number

        # If not, finish up and exit the loop.
        else:
            trials[np.isnan(trials)] = trial_number - 1
            break

        # The start of the next trial is the end of the last.
        trial_start = trial_end

    # Debugging purposes.
    # for trial in range(int(max(trials))):
    #     plt.plot(position[trials == trial])

    return trials.astype(int)


def spiral_plot(behavior_df, markers, marker_legend='Licks'):
    """
    Plot trajectory of the mouse over time in a circular (polar) axis. Theta
    corresponds to the animal's position while the radius (distance from center)
    is time. Also plot events of interest (e.g., licks or calcium activity).

    :parameters
    ---
    behavior_df: DataFrame
        From either Preprocess or Session.

    markers: array
        Something that indexes behavior_df. These locations will be highlighted.

    marker_legend: str
        Label for whatever you are highlighting
    """
    position = np.asarray(linearize_trajectory(behavior_df)[0])
    t = np.asarray(behavior_df.frame)
    #frame_rate = 30
    #t_labels = np.linspace(0, max(t), 4) / frame_rate

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(position, t)
    ax.plot(position[markers], t[markers], 'ro', markersize=2)
    ax.legend(['Trajectory', marker_legend])

    # Clean up axes.
    ax.spines['polar'].set_visible(False)
    ax.set_xticklabels([])
    #ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_theta_zero_location("N")     # Make 12 o'clock "0 degrees".
    ax.set_theta_direction(-1)          # Polar coordinates go counterclockwise.
    #ax.set_yticklabels(t_labels)

    return fig, ax


def approach_speed(behavior_df, location, window=(-30, 30), dist_thresh=0.03,
                   smoothing_factor=4, ax=None, plot=True, acceleration=True):
    """
    Get the approach speed of the mouse to an arbitrary location on the maze
    (specified by location, in radians). Basically looks at the speed or
    acceleration within a window of frames surrounding the frame when the mouse
    comes within dist_thresh of the location.

    :parameters
    ---
    location: numeric
        Target location per trial.

    window: tuple
        (frames behind, frames ahead)

    dist_thresh: numeric
        Vicinity to location to set window.

    smoothing_factor: numeric
        Amount to smooth velocity or accleration trace.

    ax: Axes object
        Axis to plot on.

    plot: boolean
        Whether to plot results or just return data.

    acceleration: boolean
        Whether to look at acceleration rather than velocity.

    :return
    ---
    approaches: (trials, frame) array
        Velocity or accleration at each frame.
    """
    # Get some basic characteristics of requested parameters.
    window = np.asarray(window)
    window_size = sum(abs(window))
    ntrials = max(behavior_df['trials'] + 1)
    nframes = len(behavior_df)
    frame_rate = 30

    # Convert things into arrays.
    location = np.asarray(location).T
    mouse_location = np.asarray(behavior_df['lin_position'])
    trials = np.asarray(behavior_df['trials'])

    # Get speeds (roughly distances between samples for now) then smooth.
    if smoothing_factor > 0:
        speeds = gaussian_filter1d(np.asarray(behavior_df['distance']), smoothing_factor)
    else:
        speeds = np.asarray(behavior_df['distance'])

    # Can also get acceleration of the mouse.
    if acceleration:
        speeds = np.insert(np.diff(speeds), 0, 0)

    # For each trial, find the point in time when the mouse comes some distance
    # (dist_thresh) of the target. Then look at the velocity within a window of
    # time around that timepoint.
    approaches = nan_array((ntrials, window_size))
    dists = abs(mouse_location - location)
    at_port = dists < dist_thresh
    for trial in range(ntrials):
        there_now = np.logical_and(at_port, trials == trial)

        # Handles cases where the mouse didn't visit the target location.
        # This sometimes happens at the beginning or end of the session.
        if not any(there_now):
            continue

        # Find time zero (when mouse arrived at location). argmax finds the
        # first value where there_now is True. Also get the window.
        t0 = np.argmax(there_now)
        pre = t0 + window[0]
        post = t0 + window[1]

        # Handles edge cases where the window extends past the session start
        # or end. In those cases, pad with nans.
        front_pad = 0
        back_pad = 0
        if pre < 0:
            front_pad = 0 - pre
            pre = 0
        if post > nframes:
            back_pad = post - nframes
            post = nframes

        # Index and insert into preallocated array.
        window_ind = slice(pre, post)
        to_insert = speeds[window_ind]
        to_insert = np.pad(to_insert, (front_pad, back_pad), mode='constant',
                           constant_values=np.nan)
        approaches[trial] = to_insert

    if plot:
        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(approaches, cmap='bwr')
        ax.axis('tight')
        ax.axvline(x=window_size / 2, color='r')  # t0 line
        ax.set_xticks([0, window_size / 2, window_size - 1])
        ax.set_xticklabels([np.round(window[0] / frame_rate, 1),
                            0,
                            np.round(window[1] / frame_rate, 1)])
        ax.set_xlabel('Time to reach location (s)')
        ax.set_ylabel('Trials')

    return approaches


def blocked_approach_speeds(approaches, blocks=4, plot=True, ax=None,
                            cmap='copper'):
    """
    NOTE: Not very informative.
    Splits the session into equal blocks and plots the mean of the velocities
    or accelerations.


    :parameters
    ---
    approaches: (trials, frames) array
        From approach_speeds.

    blocks: int
        Number of blocks to split into.

    plot: boolean
        Whether or not to plot.

    ax: Axes object
        Axis to plot on.

    cmap: str
        Colormap to plot lines with.

    :return
    ---
    split_approaches: list of (trials, frames) arrays
        Session split into equal blocks.

    """

    split_approaches = np.array_split(approaches, blocks)
    cmap = plt.get_cmap(cmap)

    if plot:
        if ax is None:
            fig, ax = plt.subplots()

        # Follow a defined sequential color map so we know the progression.
        ax.set_prop_cycle('color', cmap(np.linspace(0, 1, blocks)))

        for approach in split_approaches:
            ax.plot(np.nanmean(approach, axis=0))

    return split_approaches


class Preprocess:
    def __init__(self, folder=None, sync_mode='frame'):
        """
        Preprocesses behavior data by specifying a session folder.

        :parameter
        ---
        folder: str
            Folder path to session.
        """
        if folder is None:
            self.folder = filedialog.askdirectory()
        else:
            self.folder = folder

        # Get the paths to relevant files.
        self.paths = grab_paths(self.folder)
        self.paths['PreprocessedBehavior'] = \
            os.path.join(self.folder, 'PreprocessedBehavior.csv')

        # Check if Preprocess has been ran already by attempting
        # to load a pkl file.
        try:
            self.behavior_df  = pd.read_csv(self.paths['PreprocessedBehavior'])

        # If not, sync Arduino data.
        except:
            try:
                convert_dlc_to_eztrack(self.paths['DLC'])
                print('DeepLabCut .h5 successfully converted to .csv.')
                self.paths.update(grab_paths(self.folder))
            except:
                print('DLC file not detected. Reading ezTrack file instead.')

            self.behavior_df = sync_Arduino_outputs(self.paths['Arduino'],
                                                    self.paths['BehaviorData'],
                                                    sync_mode=sync_mode)[0]

            # Find timestamps where the mouse seemingly teleports to a new location.
            # This is likely from mistracking. Interpolate those data points.
            self.interp_mistracks()

            #self.behavior_df = clean_lick_detection(self.behavior_df)
            self.preprocess()

    def preprocess(self):
        """
        Fill in DataFrame columns with some calculated values:
            Linearized position
            Trial number
            Distance (velocity)

        """
        self.behavior_df['lin_position'] = linearize_trajectory(self.behavior_df)[0]
        self.behavior_df['trials'] = get_trials(self.behavior_df)
        self.behavior_df['distance'] = consecutive_dist(np.asarray((self.behavior_df.x,
                                                                    self.behavior_df.y)).T,
                                                        zero_pad=True)


    def save(self, path=None, fname='PreprocessedBehavior.csv'):
        """
        Save preprocessed data.

        path: str
            Folder path to save to. If None, default to session folder.

        fname: str
            File name to call the pkl file.

        """
        if path is None:
            fpath = self.paths['PreprocessedBehavior']
        else:
            fpath = os.path.join(path, fname)

        self.behavior_df.to_csv(fpath, index=False)


    def interp_mistracks(self, thresh=4):
        """
        Z-score the velocity and find abnormally fast movements. Interpolate those.

        :parameter
        ---
        thresh: float
            Number of standard deviations above the mean to be called a mistrack.
        """
        mistracks = zscore(self.behavior_df['distance']) > thresh
        self.behavior_df.loc[mistracks, ['x','y']] = np.nan
        self.behavior_df.interpolate(method='linear', columns=['x', 'y'],
                                     inplace=True)


    def plot_frames(self, frame_number):
        """
        Plot frame and position from ezTrack csv.

        :parameter
        frame_num: int
            Frame number that you want to start on.
        """
        vid = cv2.VideoCapture(self.paths['BehaviorVideo'], cv2.CAP_FFMPEG)
        n_frames = int(vid.get(7))
        frame_nums = ["Frame " + str(n) for n in range(n_frames)]
        self.f = ScrollPlot(disp_frame,
                            current_position=frame_number,
                            vid_fpath=self.paths['BehaviorVideo'],
                            x=self.behavior_df['x'], y=self.behavior_df['y'],
                            titles=frame_nums)


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
            start_frame = 0

        # Plot frame and position, then connect to mouse.
        self.plot_frames(start_frame)
        self.f.fig.canvas.mpl_connect('button_press_event',
                                      self.correct)

        # Wait for click.
        while plt.get_fignums():
            plt.waitforbuttonpress()

        self.preprocess()


    def correct(self, event):
        """
        Defines what happens during mouse clicks.

        :parameter
        ---
        event: click event
            Defined by mpl_connect. Don't modify.
        """
        # Overwrite DataFrame with new x and y values.
        self.behavior_df.loc[self.f.current_position, 'x'] = event.xdata
        self.behavior_df.loc[self.f.current_position, 'y'] = event.ydata

        # Plot the new x and y.
        self.f.fig.axes[0].plot(event.xdata, event.ydata, 'go')
        self.f.fig.canvas.draw()


    def find_outliers(self):
        """
        Plot the distances between points and then select with your
        mouse where you would want to do a manual correction from.

        """
        # Plot distance between points and connect to mouse.
        # Clicking the plot will bring you to the frame you want to
        # correct.
        self.traj_fig, self.traj_ax = plt.subplots(1,1,num='outliers')
        self.dist_ax = self.traj_ax.twinx()

        # Re-do linearize trajectory and velocity calculation.
        radii = linearize_trajectory(self.behavior_df)[1]
        self.behavior_df['distance'][1:] = \
            consecutive_dist(np.asarray((self.behavior_df.x, self.behavior_df.y)).T,
                             axis=0)

        # Plot.
        self.traj_ax.plot(radii, alpha=0.5)
        self.traj_ax.set_ylabel('Distance from center', color='b')
        self.dist_ax.plot(self.behavior_df['distance'], color='r', alpha=0.5)
        self.dist_ax.set_ylabel('Velocity', color='r', rotation=-90)
        self.dist_ax.set_xlabel('Frame')
        self.traj_fig.canvas.mpl_connect('button_press_event',
                                          self.jump_to)

        while plt.fignum_exists('outliers'):
            plt.waitforbuttonpress()


    def jump_to(self, event):
        """
        Jump to this frame based on a click on a graph. Grabs the x (frame)
        """
        plt.close(self.traj_fig)
        if event.xdata < 0:
            event.xdata = 0
        self.correct_position(int(np.round(event.xdata)))


    def plot_lin_position(self):
        """
        Plots the linearized position for the whole session, color-coded by trial.

        """
        for trial in range(int(max(self.behavior_df['trials']))):
            plt.plot(self.behavior_df['lin_position'][self.behavior_df['trials'] == trial])


    def plot_trial(self, trial):
        """
        Plots any trial (non-linearized).

        :parameter
        ---
        trial: int
            Trial number
        """
        x = self.behavior_df['x']
        y = self.behavior_df['y']
        idx = self.behavior_df['trials'] == trial

        fig, ax = plt.subplots()
        ax.plot(x[idx], y[idx])
        ax.set_aspect('equal')


    def track_video(self):
        make_tracking_video(self.paths['BehaviorVideo'], self.paths['BehaviorData'],
                            Arduino_path=self.paths['Arduino'])


class Session:
    def __init__(self, folder=None):
        """
        Contains many useful analyses for single session data.

        :parameter
        ---
        folder: str
            Directory pertaining to a single session. If not specified, a
            dialog box will appear for you to navigate to the folder.

        Useful methods:
            plot_licks(): Plots the trajectory and licks in a spiral pattern.

            port_approaches(): Plots speed or acceleration within a time
                window centered around the arrival to each port.

            sdt_trials(): Gets the hit/miss/false alarm/correct rejection rate
                for each port. Currently in beta.

        """
        # If folder is not specified, open a dialog box.
        if folder is None:
            self.folder = filedialog.askdirectory()
        else:
            self.folder = folder

        # Find paths.
        self.paths = grab_paths(self.folder)
        self.paths['PreprocessedBehavior'] = \
            os.path.join(self.folder, 'PreprocessedBehavior.csv')

        # Try loading a presaved csv.
        try:
            self.behavior_df = pd.read_csv(self.paths['PreprocessedBehavior'])
        except:
            raise FileNotFoundError('Run Preprocess() first.')

        # Number of laps run.
        self.ntrials = max(self.behavior_df['trials'] + 1)

        # Amount of time spent per trial (in frames).
        self.frames_per_trial = np.bincount(self.behavior_df['trials'])

        # Find water ports.
        self.ports, self.lin_ports = find_water_ports(self.behavior_df)

        # Find rewarded ports
        rewarded_ports = find_rewarded_ports(self.behavior_df)
        self.rewarded = np.zeros(8, dtype=bool)
        self.rewarded[rewarded_ports] = True
        self.n_rewarded = np.sum(self.rewarded)


    def plot_licks(self):
        """
        Plot the trajectory and licks in a spiral pattern (polar plot).

        """
        fig, ax = spiral_plot(self.behavior_df, self.behavior_df['lick_port'] > -1)

        for port in self.lin_ports:
            ax.axvline(x=port, color='g')

        ax.legend(['Trajectory', 'Licks', 'Ports'])


    def port_approaches(self, window=(-15, 15), dist_thresh=0.03,
                        smoothing_factor=4,  acceleration=True, plot=True):
        """
        Plots the speed or acceleration of the mouse to each port per trial.

        :parameters
        ---
        window: tuple
            (frames behind, frames ahead) that you want to analyze.

        dist_thresh: numeric
            When the mouse first gets within this many distance units to the port,
            that will be considered when it has arrived and the window will be
            centered on that frame.

        smoothing_factor: int
            Amount to smooth velocity/acceleration by.

        acceleration: bool
            Whether to compute acceleration (when True) or speed (when False).

        plot: bool
            Whether or not to plot.

        """
        self.approaches = []
        self.window = window

        if plot:
            fig = plt.figure(figsize=(6, 8.5))
        else:
            fig = None

        for i, port in enumerate(self.lin_ports):
            if plot:
                ax = fig.add_subplot(4, 2, i + 1)
            else:
                ax = None
            approaches = approach_speed(self.behavior_df, port, window=window,
                                        dist_thresh=dist_thresh,
                                        smoothing_factor=smoothing_factor,
                                        acceleration=acceleration,
                                        plot=plot, ax=ax)
            self.approaches.append(approaches)

        if plot:
            fig.tight_layout()
            max_speed = np.max([np.nanmax(approach) for approach in self.approaches])
            min_speed = np.min([np.nanmin(approach) for approach in self.approaches])
            for ax in fig.axes:
                for im in ax.get_images():
                    im.set_clim(min_speed, max_speed)


    def blocked_port_approaches(self, blocks=4):
        """
        NOTE: Not very informative.

        :param blocks:
        :return:
        """
        if not hasattr(self, 'approaches'):
            self.port_approaches(plot=False)
        fig = plt.figure(figsize=(6, 8.5), num='blocked_approaches')
        for i, approach in enumerate(self.approaches):
            try:
                ax = fig.add_subplot(4, 2, i + 1, sharey=ax)
            except:
                ax = fig.add_subplot(4, 2, i + 1)

            blocked_approach_speeds(approach, blocks=blocks, ax=ax)

        fig.tight_layout()

        pass


    def get_licks(self, plot=True):
        """
        Plots the number of licks on each port per trial.

        :parameter
        ---
        plot: boolean
            Whether or not to plot.
        """
        ports = range(8)
        all_licks = []
        for trial in range(self.ntrials):
            this_trial = self.behavior_df['trials'] == trial
            licks = list(self.behavior_df.loc[this_trial, 'lick_port'])

            licks_per_port = [licks.count(port) for port in ports]

            all_licks.append(licks_per_port)

        if plot:
            fig, ax = plt.subplots(figsize=(4.35, 5))
            ax.imshow(all_licks)
            ax.axis('tight')
            ax.set_xlabel('Water port #')
            ax.set_ylabel('Trial')

        return all_licks


    def sdt_trials(self, blocks=None, plot=True):
        # Get number of licks per port.
        self.all_licks = np.asarray(self.get_licks(plot=False))

        # Split the session into N blocks.
        if blocks is not None:
            licks = np.array_split(self.all_licks, blocks)
        else:
            licks = [self.all_licks]

        # Preallocate dict.
        sdt = {'hits': [],
               'misses': [],
               'FAs': [],
               'CRs': []
               }
        for trial_block in licks:
            # Get number of passes through ports that should or should not be licked.
            ntrials = len(trial_block)
            go_trials = ntrials * self.n_rewarded
            nogo_trials = ntrials * (8-self.n_rewarded)

            # Binarized the lick array so that at least one lick will mark it as
            # correct.
            binarized = trial_block > 0
            correct_licks = np.sum(binarized[:, self.rewarded])
            incorrect_licks = np.sum(binarized[:, ~self.rewarded])

            # Get rates for hits, misses, false alarms, and correct rejections.
            hit_rate = correct_licks / go_trials
            miss_rate = (go_trials - correct_licks) / go_trials
            FA_rate = incorrect_licks / nogo_trials
            CR_rate = (nogo_trials - incorrect_licks) / nogo_trials

            sdt['hits'].append(hit_rate)
            sdt['misses'].append(miss_rate)
            sdt['FAs'].append(FA_rate)
            sdt['CRs'].append(CR_rate)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(sdt['hits'])
            ax.plot(sdt['CRs'])
            ax.legend(('Hits', 'Correct rejections'))

        return sdt

    def SDT(self):
        """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
        # Floors an ceilings are replaced by half hits and half FA's
        sdt = self.sdt_trials(blocks=4)
        Z = norm.ppf

        d_prime = []
        for hits, misses, fas, crs in zip(sdt['hits'],
                                          sdt['misses'],
                                          sdt['FAs'],
                                          sdt['CRs']):
            half_hit = 0.5 / (hits + misses)
            half_fa = 0.5 / (fas + crs)

            # Calculate hit_rate and avoid d' infinity
            hit_rate = hits / (hits + misses)
            if hit_rate == 1:
                hit_rate = 1 - half_hit
            if hit_rate == 0:
                hit_rate = half_hit

            # Calculate false alarm rate and avoid d' infinity
            fa_rate = fas / (fas + crs)
            if fa_rate == 1:
                fa_rate = 1 - half_fa
            if fa_rate == 0:
                fa_rate = half_fa

            # Return d'
            d_prime.append(Z(hit_rate) - Z(fa_rate))

        return d_prime


if __name__ == '__main__':
    #folder = r'D:\Projects\CircleTrack\Mouse4\01_30_2020\H16_M50_S22'
    # folder = r'D:\Projects\CircleTrack\Mouse4\02_01_2020\H15_M37_S17'
    # data = Session(folder)
    #data.save()
    #data = Session(folder)
    #data.plot_licks()

    from CircleTrack.sql import Database
    with Database() as db:
        mouse_id = db.conditional_ID_query('mouse', 'id', 'name', 'Mouse4')[0]
        paths = db.conditional_ID_query('session', 'path', 'mouse_id', mouse_id)

    d = []
    for path in paths:
        data = Session(path)
        d.append(data.SDT())

    plt.plot(d[16:], '.-')
    for x in np.arange(3.5, 11.5, 4):
        plt.axvline(x=x, color='r')
    for x in np.arange(11.5, 18.5, 4):
        plt.axvline(x=x, color='magenta')

    plt.ylabel('d prime')
    plt.xlabel('Days')
    plt.xlim([-0.5, 19.5])
    labels = np.arange(1, 6)
    positions = np.arange(1.5, 21.5, 4)
    plt.xticks(positions, labels)

    pass