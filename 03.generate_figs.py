# %% imports
import pickle as pkl
import warnings

# from scipy.stats import wilcoxon, mannwhitneyu, ttest_ind
import numpy as np
import pingouin as pg

from routine.CircleTrack.RecentReversal import RecentReversal

warnings.filterwarnings("ignore")

# IN_ANM = [
#         "Fornax",
#         "Gemini",
#         "Janus",
#         "Lyra",
#         "Miranda",
#         "Naiad",
#         "Oberon",
#         "Puck",
#         #"Rhea",
#         "Sao",
#         "Titania",
#         "Umbriel",
#         "Virgo",
#         "Ymir",
#         "Atlas"
#     ]

IN_ANM = ["Janus", "PhantomLyra"]
IN_PV_CORR = "./intermediate/pv_corr.pkl"
IN_DB = "./intermediate/database.sqlite"
FIG_PATH = "./figs/publish"

# %%
# For R code.
import os

# os.environ['R_HOME'] = r"C:\Users\wm228\Anaconda3\envs\ca_imaging\Lib\R"  #Replace with your R directory.
# os.environ["PATH"]   = r"C:\Users\wm228\Anaconda3\Lib\R" + ";" + os.environ["PATH"]
# import rpy2.robjects as robjects

# %%
# Get all session data.

RR = RecentReversal(
    IN_ANM,
    db_fname=IN_DB,
    project_name="RemoteReversal",
    behavior_only=False,
    save_path=FIG_PATH,
)
with open(IN_PV_CORR, "rb") as pvf:
    pv_corr = pkl.load(pvf)

# %%
from routine.CircleTrack.Chemogenetics import Chemogenetics

# %%
PSAM_mice = ["PSAM_" + str(i) for i in np.arange(4, 28)]
mistargets = ["PSAM_" + str(i) for i in [4, 5, 6, 8]]
PSAM_mice.remove("PSAM_18")  # Bad reversal session.
exclude_non_learners = True
if exclude_non_learners:
    [PSAM_mice.remove(x) for x in ["PSAM_" + str(i) for i in [13, 15]]]
[PSAM_mice.remove(x) for x in mistargets]
C = Chemogenetics(PSAM_mice, actuator="PSAM")

# %% [markdown]
# # Figure 1.

# %% [markdown]
# ## A. Schematic of spatial reversal task

# %% [markdown]
# ## B-F. Behavior

# %%
anova_df, pairwise_df = RR.make_fig1(panels="B", corr_matrices=pv_corr)
print(anova_df)
print("")
print(pairwise_df.loc[pairwise_df["p-corr"] < 0.05])

# %%
anova_df, pairwise_df = RR.make_fig1(panels="C", corr_matrices=pv_corr)
print(anova_df)
print("")
print(pairwise_df.loc[pairwise_df["p-corr"] < 0.05])

# %%
anova_df, pairwise_df = RR.make_fig1(panels="D", corr_matrices=pv_corr)
print(anova_df)
print("")
print(pairwise_df.loc[pairwise_df["p-corr"] < 0.05])

# %%
RR.make_fig1(panels="E", corr_matrices=pv_corr)

# %%
RR.make_fig1(panels="F", corr_matrices=pv_corr)

# %% [markdown]
# ## H-L. Basic calcium imaging data

# %%
RR.make_fig1(panels="H", corr_matrices=pv_corr, mouse="Janus")

# %%
n_neurons = RR.make_fig1(panels="J", corr_matrices=pv_corr)

# %%
RR.make_fig1(panels="K", corr_matrices=pv_corr, mouse="Janus")

# %%
RR.make_fig1(panels="L", corr_matrices=pv_corr, mouse="Janus")

# %%
# Decoding takes hours.
# RR.make_fig1('M')

# %% [markdown]
# ## N-S. Lack of remapping

# %%
data, df = RR.make_fig1(corr_matrices=pv_corr, panels="N")

# %%
rhos = RR.make_fig1(corr_matrices=pv_corr, panels="O")

# %%
rhos = RR.make_fig1(corr_matrices=pv_corr, panels="P")

# %%
remap_score_df = RR.make_fig1(corr_matrices=pv_corr, panels="Q")

# %%
remap_score_df = RR.make_fig1(corr_matrices=pv_corr, panels="R")

# %%
remap_score_df = RR.make_fig1(corr_matrices=pv_corr, panels="S")

# %% [markdown]
# # Figure 2.

# %% [markdown]
# ## A-D. Example ensembles

# %%
RR.make_fig2("A", mouse="PhantomLyra")

# %%
RR.make_fig2("B", mouse="PhantomLyra")

# %%
RR.make_fig2("C", mouse="PhantomLyra")

# %%
n_ensembles = RR.make_fig2("D")

# %% [markdown]
# ## E-G. Ensemble decoding

# %%
# Takes a while.
anova_df, pairwise_df, errors_df = RR.make_fig2("E")

# %%
RR.make_fig2("F", mouse="PhantomLyra")

# %%
df = RR.make_fig2("G")

# %%
anova_df = pg.rm_anova(
    df,
    dv="scores",
    within=["decoded_session", "time_bins"],
    subject="mice",
)
anova_df

# %%
pairwise_dfs = {
    session: pg.pairwise_ttests(
        dv="scores",
        within="time_bins",
        subject="mice",
        data=df[df["decoded_session"] == session],
        padjust="fdr_bh",
    )
    for session in np.unique(df["decoded_session"].values)
}
pairwise_dfs

# %% [markdown]
# # Figure 3

# %% [markdown]
# ## A-B. Remodeling and non-remodeling ensembles

# %%
RR.make_fig3("A", mouse="PhantomLyra")

# %%
RR.make_fig3("B", mouse="PhantomLyra")

# %% [markdown]
# ## C-E. Relationship of remodeling ensembles to behavior

# %%
RR.make_fig3("C")

# %%
RR.make_fig3("D")

# %%
RR.make_fig3("E")

# %% [markdown]
# ## F-G. Intraconnectivity of remodeling ensembles

# %%
RR.make_fig3("F", mouse="Janus")

# %%
anova_dfs = RR.make_fig3("G")

# %%
for df in anova_dfs.values():
    display(df)

# %% [markdown]
# # Figure 4

# %%
anova_dfs = RR.make_fig4("A")

# %%
for df in anova_dfs.values():
    display(df)

# %%
RR.make_fig4("B", mouse="Janus")

# %%
RR.make_fig4("C", mouse="Janus")

# %%
Degrees = RR.make_fig4("D")

# %% [markdown]
# For linear mixed model statistics, run R code below, or lines `1:12` in `will_analysis.R` from https://github.com/jetsetbaxter/willmau.

import rpy2.robjects.packages as rpackages

# %%
# Make sure you have emmeans package.
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)
packnames = ("emmeans", "Matrix", "lme4")
utils.install_packages(StrVector(packnames))

# %%
# %load_ext rpy2.ipython
# %R require("lmerTest")
# %R require("here")
# %R require("emmeans")
# %R require("tidyverse")

# # %%
# %%R
# degrees <- read_csv("C:/Users/wm228/Documents/GitHub/memory_flexibility/Figures/4/degrees_df.csv") %>% select(-`...1`)

# lmer(degree ~ category + (1|mouse) + (1|ensemble_id:mouse) + (1|neuron_id:mouse), data = degrees) %>% summary()

# # %%
# act_rate_df = RR.make_fig4("E")

# # %% [markdown]
# # For linear mixed model statistics, run R code below, or run lines `127:142` in `will_analysis.R` from https://github.com/jetsetbaxter/willmau.

# # %%
# %%R
# dpath = "C:/Users/wm228/Documents/GitHub/memory_flexibility/Figures/S7/all_sessions_activity_rate.csv"
# all_sessions <- read_csv(dpath) %>% select(-`...1`) %>%
#   mutate(session_id = factor(session_id,
#                              levels = c("Goals1", "Goals2", "Goals3", "Goals4", "Reversal")))

# all_sessions_model <-
#   lmer(event_rate ~ category*session_id + (1|mouse) + (1|ensemble_id:mouse) + (1|neuron_id:mouse), data = all_sessions)

# summary(all_sessions_model)
# anova(all_sessions_model)

# # these take a moment
# all_sessions_emm <-
#   all_sessions_model %>% emmeans(pairwise ~ category | session_id, pbkrtest.limit = 5000)
# all_sessions_emm
# all_sessions_emm %>% plot()

# %% [markdown]
# # Figure 5

# %%
RR.make_fig5("A")

# %%
RR.make_fig5("B")

# %%
RR.make_fig5("C")

# %%
RR.make_fig5("D")

# %%
RR.make_sfig1("A")

# %%
RR.make_sfig1("B")

# %%
RR.make_sfig1("C")

# %%
RR.make_sfig1("D")

# %%
RR.make_sfig1("E")

# %%
C.make_sfig2("A")

# %%
C.make_sfig2("C")

# %%
C.make_sfig2("D")

# %%
C.make_sfig2("E")

# %%
C.make_sfig2("F")

# %%
C.make_sfig2("G")

# %%
C.make_sfig2("H")

# %%
RR.make_sfig3()

# %%
SI_anova, pairwise_df = RR.make_sfig4("A")

# %%
ensemble_size_df = RR.make_sfig4("B")

# %%
RR.make_sfig5("A", mouse="Janus")

# %%
RR.make_sfig5("B", mouse="Janus")

# %%
RR.make_sfig5("C", mouse="Janus")

# %%
df = RR.make_sfig5("D")

# %%
anova_df = pg.rm_anova(
    df,
    dv="scores",
    within=["decoded_session", "time_bins"],
    subject="mice",
)
anova_df

# %%
pairwise_dfs = {
    session: pg.pairwise_ttests(
        dv="scores",
        within="time_bins",
        subject="mice",
        data=df[df["decoded_session"] == session],
        padjust="fdr_bh",
    )
    for session in np.unique(df["decoded_session"].values)
}
pairwise_dfs["Reversal"]

# %%
RR.make_sfig6("A")

# %%
RR.make_sfig6("B")

# %%
RR.make_sfig6("C")

# %%
RR.make_sfig7("A")

# %%
RR.make_sfig7("B")

# %%
RR.make_sfig7("C")

# %%
# Takes forever.
# RR.make_sfig8("A")

# %%
# Takes forever.
# RR.make_fig8("B")

# %%
RR.make_sfig9("A")

# %%
RR.make_sfig9("B")

# %%
RR.make_sfig9("C")

# %%
RR.make_sfig9("D")

# %%
RR.make_sfig9("E")

# %%
RR.make_sfig9("F")

# %%
RR.make_sfig9("G")

# %%
RR.make_sfig9("H")
