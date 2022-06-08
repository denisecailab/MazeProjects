#%% imports and definitions
import os
from routine.CircleTrack.MiniscopeFunctions import CalciumSession
from routine.util import walklevel

IN_DPATH = "./data"

#%% process calcium
for root, dirs, files in walklevel(IN_DPATH, depth=3):
    if "Miniscope" in dirs:
        try:
            os.remove(os.path.join(root, "metadata.pkl"))
        except OSError:
            pass
        print("processing {}".format(root))
        C = CalciumSession(
            root,
            overwrite_assemblies=True,
            overwrite_placefields=True,
            overwrite_synced_data=True,
            overwrite_placefield_trials=True,
        )
