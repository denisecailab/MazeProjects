#%% imports and definitions
import os

from routine.util import walklevel

IN_DPATH = "./data"

#%% preprocess behaviors
for root, dirs, files in walklevel(IN_DPATH, depth=3):
    if "Miniscope" in dirs:
        try:
            os.remove(os.path.join(root, "metadata.pkl"))
        except OSError:
            pass
