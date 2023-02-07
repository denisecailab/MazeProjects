#%% imports and definitions
import os

import numpy as np
import pandas as pd
import xarray as xr

from routine.CircleTrack.RecentReversal import RecentReversal

IN_ANM = [
    "Fornax",
    "Gemini",
    "Janus",
    "Lyra",
    "Miranda",
    "Naiad",
    "Oberon",
    "Puck",
    # "Rhea",
    "Sao",
    "Titania",
    "Umbriel",
    "Virgo",
    "Ymir",
    "Atlas",
]
IN_DB = "./intermediate/database.sqlite"
IN_SS_DICT = {
    "Goals1": "Training1",
    "Goals2": "Training2",
    "Goals3": "Training3",
    "Goals4": "Training4",
    "Reversal": "Reversal1",
}
FIG_PATH = "./figs/publish"
OUT_PATH = "./output/"
DRY_RUN = False
os.makedirs(OUT_PATH, exist_ok=True)

#%% load data
RR = RecentReversal(
    IN_ANM,
    db_fname=IN_DB,
    project_name="RemoteReversal",
    behavior_only=False,
    save_path=FIG_PATH,
)
#%% extract data
data = RR.data
imag_path = os.path.join(OUT_PATH, "processed")
behav_path = os.path.join(OUT_PATH, "behav")
assemb_path = os.path.join(OUT_PATH, "assemblies")
reg_path = os.path.join(OUT_PATH, "registration")
os.makedirs(imag_path, exist_ok=True)
os.makedirs(behav_path, exist_ok=True)
os.makedirs(assemb_path, exist_ok=True)
os.makedirs(reg_path, exist_ok=True)
n_neuron_df = pd.DataFrame(columns=["animal", "session", "nn"])
for anm, anm_dat in data.items():
    for ss, ss_dat in anm_dat.items():
        if ss in IN_SS_DICT:
            # parse data
            ss_new = IN_SS_DICT[ss]
            imag = ss_dat.imaging
            behav = ss_dat.behavior
            assemb = ss_dat.assemblies
            # parse behavior
            behav_df = behav.data["df"]
            behav_df["animal"] = anm
            behav_df["session"] = ss_new
            behav_df["cohort"] = "cohort0"
            fm_crd = np.array(behav_df["frame"])
            # parse imaging
            C = xr.DataArray(
                imag["C"],
                dims=["unit_id", "frame"],
                coords={
                    "unit_id": np.arange(imag["n_neurons"]),
                    "frame": fm_crd,
                    "frame_org": ("frame", imag["frames"]),
                },
                name="C",
            )
            S = xr.DataArray(
                imag["S"],
                dims=["unit_id", "frame"],
                coords={
                    "unit_id": np.arange(imag["n_neurons"]),
                    "frame": fm_crd,
                    "frame_org": ("frame", imag["frames"]),
                },
                name="S",
            )
            S_bin = xr.DataArray(
                imag["S_binary"],
                dims=["unit_id", "frame"],
                coords={
                    "unit_id": np.arange(imag["n_neurons"]),
                    "frame": fm_crd,
                    "frame_org": ("frame", imag["frames"]),
                },
                name="S_bin",
            )
            imag_ds = xr.merge([C, S, S_bin]).assign_coords(
                animal=anm, session=ss_new, cohort="cohort0"
            )
            n_neuron_df = pd.concat(
                [
                    n_neuron_df,
                    pd.DataFrame(
                        [{"animal": anm, "session": ss_new, "nn": imag["n_neurons"]}]
                    ),
                ],
                ignore_index=True,
            )
            # parse assemblies
            pca = assemb["significance"]
            ncomp = pca.n_components_
            nasmb = pca.nassemblies
            pca_comp = xr.DataArray(
                pca.components_,
                dims=["comp_id", "unit_id"],
                coords={
                    "comp_id": np.arange(ncomp),
                    "unit_id": np.arange(imag["n_neurons"]),
                },
                name="pca_comp",
            )
            pat = xr.DataArray(
                assemb["patterns"],
                dims=["asmb_id", "unit_id"],
                coords={
                    "asmb_id": np.arange(nasmb),
                    "unit_id": np.arange(imag["n_neurons"]),
                },
                name="patterns",
            )
            act = xr.DataArray(
                assemb["activations"],
                dims=["asmb_id", "frame"],
                coords={"asmb_id": np.arange(nasmb), "frame": fm_crd},
                name="activations",
            )
            asmb_ds = xr.merge([pca_comp, pat, act]).assign_coords(
                {"animal": anm, "session": ss_new, "cohort": "cohort0"}
            )
            # save data
            if not DRY_RUN:
                imag_ds.to_netcdf(
                    os.path.join(imag_path, "{}_{}.nc".format(anm, ss_new))
                )
                asmb_ds.to_netcdf(
                    os.path.join(assemb_path, "{}_{}.nc".format(anm, ss_new))
                )
                behav_df.to_feather(
                    os.path.join(behav_path, "{}_{}.feat".format(anm, ss_new))
                )
        elif ss == "CellReg":
            reg_df = (
                ss_dat.map.rename(columns=lambda c: c[11:])
                .rename(columns={"RecentReversal": "Reversal"})
                .rename(columns=IN_SS_DICT)
                .replace(-9999, np.nan)
            )
            for ss in reg_df.columns:
                assert (
                    len(reg_df[ss].dropna().unique())
                    <= n_neuron_df.set_index(["animal", "session"])["nn"].loc[anm, ss]
                )  # seems some cells are missing from cellreg results
            reg_df["animal"] = anm
            reg_df["cohort"] = "cohort0"
            if not DRY_RUN:
                reg_df.to_feather(os.path.join(reg_path, "{}.feat".format(anm)))
