#%% imports and definitions
import os
import re
from routine.util import walklevel
from shutil import copy2


IN_SRC = "/media/share/csstorage/Will/RemoteReversal/Data"
IN_DST = "./data/RemoteReversal"
IN_FILES = [r"PreprocessedBehavior\.csv$", r"timeStamps\.csv$", r".*\.txt$"]
DRY_RUN = False

#%% copy files
for root, dirs, files in walklevel(IN_SRC, depth=4):
    f_copy = list(filter(lambda fn: any([re.search(p, fn) for p in IN_FILES]), files))
    relpath = os.path.relpath(root, IN_SRC)
    for f in f_copy:
        src = os.path.join(root, f)
        dst = os.path.join(IN_DST, relpath, f)
        if not DRY_RUN:
            try:
                copy2(src, dst)
            except:
                print("{} failed".format(dst))
        else:
            print("copying {} to {}".format(src, dst))
