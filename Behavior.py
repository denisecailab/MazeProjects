import os

import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from LickArduino import clean_Arduino_output
from util import read_eztrack, find_closest
import matplotlib.pyplot as plt
import cv2
from CircleTrack.utils import circle_sizes, cart2pol, grab_paths

def make_tracking_video(vid_path, csv_path, output_fname='Tracking.avi',
                        start=0, stop=None, fps=30, Arduino_path=None):
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
    if Arduino_path is not None:
        eztrack = sync_Arduino_outputs(Arduino_path, csv_path)[0]
        eztrack = clean_lick_detection(eztrack)
        port_colors = ['saddlebrown',
                       'red',
                       'orange',
                       'yellow',
                       'green',
                       'blue',
                       'darkviolet',
                       'gray']
    else:
        eztrack = read_eztrack(csv_path)
        port_colors = None

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
            if Arduino_path is not None:
                licking_at_port = eztrack.at[frame_number, 'lick_port']
                if licking_at_port >= 0:
                    ax.scatter(x, y, s=200, marker='+',
                               c=port_colors[licking_at_port])

            ax.set_aspect('equal')
            plt.axis('off')

            writer.grab_frame()

            plt.cla()



def sync_Arduino_outputs(Arduino_fpath, eztrack_fpath, behav_cam=2):
    """
    This function is meant to be used in conjunction with the above
    functions and Miniscope software recordings. Miniscope software
    will save videos (behavior and imaging) in a timestamped folder.

    :param folder:
    :return:
    """
    # Read the txt file generated by Arduino.
    Arduino_data, offset = clean_Arduino_output(Arduino_fpath)

    # Read the position data generated by ezTrack.
    eztrack_data = read_eztrack(eztrack_fpath)
    eztrack_data['water'] = False
    eztrack_data['lick_port'] = int(-1)

    # Get timestamping information from DAQ output.
    folder = os.path.split(eztrack_fpath)[0]
    timestamp_fpath = os.path.join(folder, 'timestamp.dat')
    try:
        DAQ_data = pd.read_csv(timestamp_fpath, sep="\s+")
    except:
        raise Exception('DAQ timestamp.dat not found or corrupted.')

    # Only take the rows corresponding to the behavior camera.
    DAQ_data = DAQ_data[DAQ_data.camNum == behav_cam]

    # Find the frame number associated with the timestamp of a lick.
    sysClock = np.asarray(DAQ_data.sysClock)
    for i, row in Arduino_data.iterrows():
        closest_time = find_closest(sysClock, row.Timestamp - offset,
                                    sorted=True)[1]
        frame_num = DAQ_data.loc[DAQ_data.sysClock == closest_time]['frameNum']
        val = row.Data

        if val.isnumeric():
            eztrack_data.at[frame_num, 'lick_port'] = int(val)
        elif val == 'Water':
            eztrack_data.at[frame_num, 'water'] = True

    return eztrack_data, Arduino_data


def find_water_ports(eztrack_data):
    """
    Use the x and y extrema to locate water port locations. Requires that the
    maze be positioned so that a port is at the 12 o'clock position. Which port
    is not important -- the code can be modified for any orientation.

    :parameter
    ---
    eztrack_data: cleaned DataFrame from sync_Arduino_outputs()

    :return
    ---
    ports: DataFrame
        DataFrame with 'x' and 'y' columns corresponding to x and y positions of
        each water port.
    """
    (width, height, radius, center) = circle_sizes(eztrack_data.x, eztrack_data.y)
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

    # Debugging purposes.
    # port_colors = ['saddlebrown',
    #                'red',
    #                'orange',
    #                'yellow',
    #                'green',
    #                'blue',
    #                'darkviolet',
    #                'gray']
    # plt.plot(eztrack_data.x, eztrack_data.y)
    # plt.scatter(center[0], center[1], c='r')
    # for color, (i, port) in zip(port_colors, ports.iterrows()):
    #     plt.scatter(port['x'], port['y'], c=color)
    # plt.axis('equal')

    return ports


def clean_lick_detection(eztrack_data, threshold=80):
    """
    Clean lick detection data by checking that the mouse is near the port during
    a detected lick.

    :parameters
    ---
    eztrack_data: cleaned DataFrame from sync_Arduino_outputs()

    threshold: float
        Distance threshold (in pixels) to be considered "near" the port.

    :return
    ---
    eztrack_data: cleaned DataFrame after eliminating false positives.
    """
    ports = find_water_ports(eztrack_data)

    lick_frames = eztrack_data[eztrack_data.lick_port > -1]
    for i, frame in lick_frames.iterrows():
        frame = frame.copy()
        port_num =  frame.lick_port
        frame_num = frame.frame

        distance = np.sqrt((frame.x - ports.at[port_num, 'x'])**2 +
                           (frame.y - ports.at[port_num, 'y'])**2)

        if distance > threshold:
            eztrack_data.at[frame_num, 'lick_port'] = -1

    return eztrack_data


def linearize_trajectory(eztrack_data):
    """
    Linearizes circular track trajectory.

    :parameter
    ---
    eztrack_data: output from read_eztrack()

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
    x, y = eztrack_data.x, eztrack_data.y
    (width, height, radius, center) = circle_sizes(eztrack_data.x, eztrack_data.y)

    # Convert to polar coordinates.
    angles, radii = cart2pol(x-center[0], y-center[1])

    return angles, radii

if __name__ == '__main__':
    folder = r'D:\Projects\CircleTrack\Mouse1\12_20_2019'
    paths = grab_paths(folder)
    eztrack_data = read_eztrack(paths['ezTrack'])
    linearize_trajectory(eztrack_data)