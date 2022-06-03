import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import serial.tools.list_ports
from scipy.signal import argrelextrema

from util import find_closest

# This is the default port from Will's desktop computer.
# Change as needed to correspond to connected Arduino.
# You can find the appropriate port with the Arduino IDE when you
# have an Arduino connected to the USB.
default_port = 'COM4'

def list_COMports():
    """
    Lists all viable COM ports.

    """
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        print(port)

def initialize(com_port=default_port):
    """
    Initiates handshake between Arduino and Python.
    Arduino must be plugged into USB port first.

    This function returns both the Serial object that will allow
    communication with the Arduino, and also a timestamp. After
    Python establishes connection with the serial port, the Arduino
    restarts and then begins a timer (see linear_track_lick.ino),
    which this function will read and save. Simultaneously, it will
    calculate Unix time.

    :parameter
    ---
    com_port: str, name of the port corresponding to Arduino

    :returns
    ---
    ser: Serial object, attached to the Arduino.
    t: int, timestamp sent from Arduino (milliseconds since Arduino
        restart).
    clock_time: float, Unix time retrieved right after receiving
        Arduino timestamp.
    """
    # Connect to serial port. Make sure the baud rates match.
    ser = serial.Serial(com_port, 115200)
    print(f'Arduino at {com_port} connected')

    time.sleep(3)   # Necessary for Arudino reboot.

    # Send handshake signal to Arduino. Make sure this signal matches
    # the character that the Arduino is listening for.
    ser.write('g'.encode())

    # Arduino code should send a timestamp right after it receives
    # the above signal. Read it then extract Unix time.
    t = ser.readline()
    clock_t = datetime.now()

    return ser, t, clock_t

def read_Arduino(com_port=default_port,
                 directory=r'D:\Projects\CircleTrack\SyncTest'):
    """
    Read Arduino serial writes and saves to a txt file continuously.
    Arduino must be plugged in or you will error.

    :parameters
    ---
    com_port: str, name of the port corresponding to Arduino.
    directory: str, directory name.


    """
    ser, t, clock_t = initialize(com_port)
    fname = make_timestamp_fname(directory, clock_t, t)

    # Keeps going until you interrupt with Ctrl+C.
    try:
        while True:
            # Read serial port.
            data = ser.readline()

            # If there's incoming data, write line to txt file.
            if data:
                timestamp = round((datetime.now() - clock_t).total_seconds() * 1000)
                data_str = data.decode('utf-8')
                port_and_frame = data_str.split('\r\n')[0]
                data = (port_and_frame + ', ' +
                     str(timestamp) + '\r\n').encode('utf-8')

                with open(fname, 'ab+') as file:
                    file.write(data)

    except:
        ser.close()


def make_timestamp_fname(directory, clock_t, t):
    # File name building.
    date_str = clock_t.strftime('%Y-%b-%d')
    time_string = clock_t.strftime('H%H_M%M_S%S.%f')[:-2] + ' ' + t.decode('utf-8')[:-2]
    fname = os.path.join(directory, date_str, time_string + '.txt')

    # If directory doesn't exist, make it.
    if not os.path.isdir(os.path.join(directory, date_str)):
        os.mkdir(os.path.join(directory, date_str))

    return fname


def clean_Arduino_output(fpath):
    """
    Load Arduino txt output into a clean format.

    :parameter
    ---
    fpath: str
        Full file path to Arduino txt file.

    :returns
    ---
    data: DataFrame
        DataFrame with columns Data, Frame, Timestamp.
        Data is either: "Lap", "Water", or an integer 0-7 indicating port.

    offset: int
        Offset in milliseconds between Arduino and DAQ.
    """
    data = pd.read_csv(fpath, header=None)
    data.columns = ['Data','Frame','Timestamp']

    # Sometimes there are NaNs. Correct them before converting
    # everything to integers.
    try:
        data.astype({'Frame': int, 'Timestamp': int})
    except:
        data.Frame = pd.to_numeric(data.Frame, errors='coerce')
        bad_frames = np.where(~np.isfinite(data.Frame))[0]
        print('Ignoring NaNs from these rows: ' + str(bad_frames))
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)

    # In an old Arduino sketch, the variable counting Miniscope frames
    # was a signed int type, which had a maximum value of 32767. Once
    # the frame count hit this value, the variable rolled over to negatives.
    # This code snippet corrects that (without writing to the txt file).
    # This should only affect data prior to Feb 6, 2020.

    # First find the last frame closest to 32767, the limit for signed ints.
    frame_limit = np.where(data.Frame == find_closest(data.Frame, 32767)[1])[0][-1]
    try:
        inverted_sign_detected = data.Frame[frame_limit + 1] < 0
    except:
        inverted_sign_detected = False

    # If the next frame number is negative, correct the rest.
    if inverted_sign_detected:
        print('Negative frame numbers detected. They have been corrected.')
        frames = data.Frame.copy()
        corrected_frames =  frames[frame_limit + 1:] + (32767*2)
        data.loc[frame_limit + 1:, 'Frame'] = corrected_frames

    # The Arduino also sometimes incorrectly writes the wrong timestamp.
    # If any huge spikes are detected, remove them.
    ts_spikes = argrelextrema(np.asarray(data.Timestamp),
                              np.greater, order=200)[0]
    data.drop(index=ts_spikes, inplace=True)    # Drop spikes in the timestamps.
    data.reset_index(drop=True, inplace=True)

    # Also grab the offset between Arduino and DAQ
    offset = int(os.path.split(fpath)[1][-8:-4])

    return data, offset



if __name__ == '__main__':
    read_Arduino()
    #folder = r'D:\Projects\CircleTrack\Mouse2\12_18_2019'
    #fname = glob.glob(os.path.join(folder, 'H**_M**_S**.**** ****.txt'))[0]
