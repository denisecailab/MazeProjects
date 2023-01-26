#%% imports
import os
import pickle as pkl
import warnings

from routine.CircleTrack.RecentReversal import RecentReversal

warnings.filterwarnings("ignore")

IN_ANM = os.listdir("./data/RemoteReversal")
IN_DB = "./intermediate/database.sqlite"
OUT_PV_CORR = "./intermediate/pv_corr.pkl"
#%% compute pv corr
RR = RecentReversal(
    IN_ANM, db_fname=IN_DB, project_name="RemoteReversal", behavior_only=False
)
pv_corr = RR.PV_corr_all_mice()
with open(OUT_PV_CORR, "wb") as pvf:
    pkl.dump(pv_corr, pvf)
