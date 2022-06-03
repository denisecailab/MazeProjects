#%% imports and definitions
import os
from routine.CircleTrack.sql import Database

IN_PROJ = {"RemoteReversal": "./data"}
IN_MOUSE_INFO = "./data/mouse_info.csv"
OUT_DB = "./intermediate/database.sqlite"

#%% generate database
with Database(
    projects=IN_PROJ,
    db_name=OUT_DB,
    mouse_csv=IN_MOUSE_INFO,
    from_scratch=True,
) as db:
    db.create()
