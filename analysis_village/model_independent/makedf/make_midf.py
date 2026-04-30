from makedf.makedf import *
from pyanalib.pandas_helpers import *
from makedf.util import *
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# HELPERS
# ============================================================

def _first_existing_col(df, candidates, label="column"):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find {label}. Tried: {candidates}")


def _drop_truth_particle_details(mcdf):
    """
    Drop detailed primary truth particle blocks not needed here.
    Keeps global neutrino truth and multiplicities.
    """
    top_levels = list(zip(*list(mcdf.columns)))[0]

    if "mu" in top_levels:
        mcdf = mcdf.drop("mu", axis=1, level=0)

    if "p" in top_levels:
        mcdf = mcdf.drop("p", axis=1, level=0)

    if "cpi" in top_levels:
        mcdf = mcdf.drop("cpi", axis=1, level=0)

    return mcdf


# ============================================================
# MODEL-INDEPENDENT PRESELECTION
# ============================================================

def make_midf_preselection(
    slcdf,
    det="SBND",
    applyPreselection=True,
    applyBarycenterCut=True,
    nuScoreCut=0.5,
    barycenterScoreCut=0.02,
):
    """
    Model-independent preselection.

    Base preselection:
      - slc.is_clear_cosmic == 0
      - slc.nu_score > 0.5
      - FV vertex

    Additional model-independent cut:
      - slc.barycenterFM.score > 0.02
    """

    if not applyPreselection:
        return slcdf

    slcdf = slcdf[slcdf.slc.is_clear_cosmic == 0]
    slcdf = slcdf[slcdf.slc.nu_score > nuScoreCut]
    slcdf = slcdf[InFV(df=slcdf.slc.vertex, inzback=0, det=det)]

    if applyBarycenterCut:
        bary_col = _first_existing_col(
            slcdf,
            [
                ("slc", "barycenterFM", "score", "", "", ""),
                ("slc", "barycenterFM", "score"),
            ],
            label="barycenterFM score column",
        )

        slcdf = slcdf[slcdf[bary_col] > barycenterScoreCut]

    return slcdf


# ============================================================
# OPTIONAL 1-SHOWER / 2-SHOWER TOPOLOGY SELECTION
# ============================================================

def apply_mi_pfp_topology_selection(
    slcdf,
    pfpTopology="none",
    showerTrackScoreCut=0.5,
):
    """
    Optional PFP topology selection.

    pfpTopology options:
      - "none" : no topology cut
      - "1shw" : keep only slices with exactly 1 valid shower-like PFP
      - "2shw" : keep only slices with exactly 2 valid shower-like PFPs

    Important:
      - PFPs with trackScore == 0 are removed before counting.
      - Slices are not rejected just because they had trackScore == 0 PFPs.
      - After the topology cut, only shower-like PFPs are kept.
    """

    if pfpTopology is None or pfpTopology == "none":
        return slcdf

    trk_score_col = _first_existing_col(
        slcdf,
        [
            ("pfp", "trackScore", "", "", "", ""),
            ("pfp", "trackScore"),
        ],
        label="PFP trackScore column",
    )

    # Remove invalid PFPs only
    slcdf = slcdf[slcdf[trk_score_col] > 0].copy()

    slice_levels = ["entry", "rec.slc..index"]

    for lev in slice_levels:
        if lev not in slcdf.index.names:
            raise KeyError(
                f"Cannot apply PFP topology selection: index level '{lev}' "
                f"not found. Current index names are {slcdf.index.names}"
            )

    is_shower_like = slcdf[trk_score_col] < showerTrackScoreCut

    n_shower_like = (
        is_shower_like
        .groupby(level=slice_levels)
        .sum()
        .rename(("slc", "n_shower_like_pfps", "", "", "", ""))
    )

    slcdf = slcdf.join(n_shower_like, on=slice_levels)

    nshw_col = ("slc", "n_shower_like_pfps", "", "", "", "")

    if pfpTopology == "1shw":
        slcdf = slcdf[slcdf[nshw_col] == 1]

    elif pfpTopology == "2shw":
        slcdf = slcdf[slcdf[nshw_col] == 2]

    else:
        raise ValueError(
            f"Unknown pfpTopology='{pfpTopology}'. "
            "Allowed values are: 'none', '1shw', '2shw'."
        )

    # Keep only shower-like PFPs in final dataframe
    slcdf = slcdf[slcdf[trk_score_col] < showerTrackScoreCut]

    return slcdf


# ============================================================
# BASE RECO MODEL-INDEPENDENT DF
# ============================================================

def make_midf(
    f,
    applyPreselection=True,
    applyBarycenterCut=True,
    savePfp=True,
    trackScore=0.51,
    nuScoreCut=0.5,
    barycenterScoreCut=0.02,
    pfpTopology="none",
    showerTrackScoreCut=0.5,
):
    """
    Base model-independent reco dataframe.

    By default:
      - saves all PFPs
      - applies model-independent preselection
      - keeps barycenterFM variables
      - keeps corrected flash variables

    Optional topology selection:
      - pfpTopology="1shw"
      - pfpTopology="2shw"
    """

    det = loadbranches(f["recTree"], ["rec.hdr.det"]).rec.hdr.det

    if 1 == det.unique()[0]:
        DETECTOR = "SBND"
    else:
        DETECTOR = "ICARUS"

    # --------------------------------------------------------
    # Load PFP and slice-level information
    # --------------------------------------------------------
    pfpdf = make_pfpdf(f)

    slcdf = loadbranches(
        f["recTree"],
        slcbranches + barycenterFMbranches + correctedflashbranches
    )
    slcdf = slcdf.rec

    # Drop pfochar if present
    if "pfochar" in pfpdf.columns.get_level_values(1):
        pfpdf = pfpdf.drop("pfochar", axis=1, level=1)

    # --------------------------------------------------------
    # CASE 1: save all PFPs
    # --------------------------------------------------------
    if savePfp:
        slcdf = multicol_merge(
            slcdf,
            pfpdf.reset_index(level="rec.slc.reco.pfp..index"),
            left_index=True,
            right_index=True,
            how="left",
            validate="one_to_many",
        )

        pfp_idx_col = next(
            c for c in slcdf.columns
            if c == "rec.slc.reco.pfp..index"
            or (
                isinstance(c, tuple)
                and len(c) > 0
                and c[0] == "rec.slc.reco.pfp..index"
            )
        )

        slcdf = slcdf.set_index(pfp_idx_col, append=True)

    # --------------------------------------------------------
    # CASE 2: save only primary/secondary showers
    # --------------------------------------------------------
    else:
        pfpdf = pfpdf[(pfpdf.pfp.trackScore > 0)]
        pfpdf = pfpdf[(pfpdf.pfp.shw.bestplane_energy > 0)]
        pfpdf = pfpdf[
            (pfpdf.pfp.shw.start.x != -999)
            & (pfpdf.pfp.shw.start.y != -999)
            & (pfpdf.pfp.shw.start.z != -999)
        ]
        pfpdf = pfpdf[(pfpdf.pfp.trk.len > 0)]

        # Number of showers
        nshwdf = (
            pfpdf[(pfpdf.pfp.trackScore < trackScore)]
            .groupby(level=[0, 1])
            .size()
            .to_frame("n_shws")
        )
        nshwdf.columns = pd.MultiIndex.from_tuples(
            [tuple(["slc"] + list(nshwdf.columns))]
        )

        slcdf = multicol_merge(
            slcdf,
            nshwdf,
            left_index=True,
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        slcdf["slc", "n_shws"] = slcdf["slc", "n_shws"].fillna(0)

        # Primary shower
        shwdf = (
            pfpdf[(pfpdf.pfp.trackScore < trackScore)]
            .sort_values(
                pfpdf.pfp.index.names[:-1]
                + [("pfp", "shw", "bestplane_energy", "", "", "")]
            )
            .groupby(level=[0, 1])
            .nth(-1)
        )
        shwdf = shwdf.drop("trk", axis=1, level=1)
        shwdf.columns = shwdf.columns.set_levels(["primshw"], level=0)

        slcdf = multicol_merge(
            slcdf,
            shwdf.droplevel(-1),
            left_index=True,
            right_index=True,
            how="left",
            validate="one_to_one",
        )

        # Secondary shower
        shwsecdf = (
            pfpdf[(pfpdf.pfp.trackScore < trackScore)]
            .sort_values(
                pfpdf.pfp.index.names[:-1]
                + [("pfp", "shw", "bestplane_energy", "", "", "")]
            )
            .groupby(level=[0, 1])
            .nth(-2)
        )
        shwsecdf = shwsecdf.drop("trk", axis=1, level=1)
        shwsecdf.columns = shwsecdf.columns.set_levels(["secshw"], level=0)

        slcdf = multicol_merge(
            slcdf,
            shwsecdf.droplevel(-1),
            left_index=True,
            right_index=True,
            how="left",
            validate="one_to_one",
        )

        # Tracks
        trkdf = pfpdf[(pfpdf.pfp.trackScore > trackScore)]

        ntrkdf = trkdf.groupby(level=[0, 1]).size().to_frame("n_trks")
        ntrkdf.columns = pd.MultiIndex.from_tuples(
            [tuple(["slc"] + list(ntrkdf.columns))]
        )

        slcdf = multicol_merge(
            slcdf,
            ntrkdf,
            left_index=True,
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        slcdf["slc", "n_trks"] = slcdf["slc", "n_trks"].fillna(0)

    # --------------------------------------------------------
    # Apply model-independent preselection
    # --------------------------------------------------------
    slcdf = make_midf_preselection(
        slcdf,
        det=DETECTOR,
        applyPreselection=applyPreselection,
        applyBarycenterCut=applyBarycenterCut,
        nuScoreCut=nuScoreCut,
        barycenterScoreCut=barycenterScoreCut,
    )

    # --------------------------------------------------------
    # Apply optional 1-shower / 2-shower topology
    # --------------------------------------------------------
    if savePfp:
        slcdf = apply_mi_pfp_topology_selection(
            slcdf,
            pfpTopology=pfpTopology,
            showerTrackScoreCut=showerTrackScoreCut,
        )

    return slcdf


# ============================================================
# MC TRUTH DF
# ============================================================

def make_mcnudf_mi(f, **args):
    mcdf = make_mcnudf(f, **args)
    mcdf = _drop_truth_particle_details(mcdf)
    return mcdf


def make_mcnudf_mi_selected_wgt(
    f,
    selected_slicedf,
    multisim_nuniv=100,
    genie_multisim_nuniv=100,
    wgt_types=["bnb", "genie", "g4"],
    slim=True,
    genie_systematics=None,
):
    mcdf_full = make_mcnudf_mi(
        f,
        include_weights=False,
        slim=slim,
    )

    tmatch_col = _first_existing_col(
        selected_slicedf,
        [
            ("slc", "tmatch", "idx", "", "", ""),
            ("slc", "tmatch", "idx"),
        ],
        label="slice truth match index column",
    )

    entry_col = ("entry", "", "", "", "", "")

    selected = selected_slicedf.reset_index()[
        [entry_col, tmatch_col]
    ].dropna()

    if len(selected) == 0:
        return mcdf_full.iloc[0:0].copy()

    selected_pairs_df = pd.DataFrame({
        "entry": selected[entry_col].to_numpy(dtype=int),
        "rec.mc.nu..index": selected[tmatch_col].to_numpy(dtype=int),
    }).drop_duplicates()

    selected_pairs = pd.MultiIndex.from_frame(selected_pairs_df)

    mcdf = mcdf_full[mcdf_full.index.isin(selected_pairs)].copy()

    if len(mcdf) == 0:
        return mcdf

    mcdf["ind"] = mcdf.index.get_level_values(1)

    df_list = []

    if "bnb" in wgt_types:
        # Flux weights must be built with the full neutrino index,
        # otherwise getsyst crashes due to shape mismatch.
        full_nuind = pd.Series(

            mcdf_full.index.get_level_values(1).to_numpy(dtype=int),
            index=mcdf_full.index,
            name="ind",

        )

        bnbwgtdf_full = bnbsyst.bnbsyst(
            f,
            full_nuind,
            multisim_nuniv=multisim_nuniv,
            slim=slim,
        )

        bnbwgtdf = bnbwgtdf_full.loc[
            bnbwgtdf_full.index.intersection(mcdf.index)
        ]

        df_list.append(bnbwgtdf)

    if "genie" in wgt_types:
        full_nuind = pd.Series(
            mcdf_full.index.get_level_values(1).to_numpy(dtype=int),
            index=mcdf_full.index,
            name="ind",
        )

        geniewgtdf_full = geniesyst.geniesyst(
            f,
            full_nuind,
            multisim_nuniv=genie_multisim_nuniv,
            slim=slim,
            systematics=genie_systematics,
        )

        geniewgtdf = geniewgtdf_full.loc[
            geniewgtdf_full.index.intersection(mcdf.index)
        ]

        df_list.append(geniewgtdf)

    if "g4" in wgt_types:
        g4wgtdf = g4syst.g4syst(
            f,
            mcdf.ind,
        )
        df_list.append(g4wgtdf)

    if len(df_list) > 0:
        wgtdf = pd.concat(df_list, axis=1)
        mcdf = multicol_concat(mcdf, wgtdf)

    return mcdf

# ============================================================
# RECO + MATCHED MC TRUTH
# ============================================================

def make_midf_mcnu(
    f,
    include_weights=False,
    multisim_nuniv=100,
    genie_multisim_nuniv=100,
    slim=True,
    applyPreselection=True,
    applyBarycenterCut=True,
    savePfp=True,
    nuScoreCut=0.5,
    barycenterScoreCut=0.02,
    pfpTopology="none",
    showerTrackScoreCut=0.5,
    wgt_types=["bnb", "genie", "g4"],
    genie_systematics=None,
):
    """
    Model-independent reco + matched MC neutrino truth dataframe.

    Important:
      - reco selection is applied first
      - weights are added only to selected matched MC neutrinos
    """

    slcdf = make_midf(
        f,
        applyPreselection=applyPreselection,
        applyBarycenterCut=applyBarycenterCut,
        savePfp=savePfp,
        nuScoreCut=nuScoreCut,
        barycenterScoreCut=barycenterScoreCut,
        pfpTopology=pfpTopology,
        showerTrackScoreCut=showerTrackScoreCut,
    )

    if include_weights:
        mcdf = make_mcnudf_mi_selected_wgt(
            f,
            selected_slicedf=slcdf,
            multisim_nuniv=multisim_nuniv,
            genie_multisim_nuniv=genie_multisim_nuniv,
            wgt_types=wgt_types,
            slim=slim,
            genie_systematics=genie_systematics,
        )
    else:
        mcdf = make_mcnudf_mi(
            f,
            include_weights=False,
            slim=slim,
        )

    mcdf.columns = pd.MultiIndex.from_tuples(
        [tuple(["slc", "truth"] + list(c)) for c in mcdf.columns]
    )

    df = multicol_merge(
        slcdf.reset_index(),
        mcdf.reset_index(),
        left_on=[
            ("entry", "", "", "", "", ""),
            ("slc", "tmatch", "idx", "", "", ""),
        ],
        right_on=[
            ("entry", "", "", "", "", ""),
            ("rec.mc.nu..index", "", ""),
        ],
        how="left",
    )

    df = df.set_index(slcdf.index.names, verify_integrity=True)

    return df


# ============================================================
# MC WRAPPERS FOR CONFIGS
# ============================================================

def make_midf_mcnu_preselect_savepfp(f):
    return make_midf_mcnu(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        include_weights=False,
        pfpTopology="none",
    )


def make_midf_mcnu_nopreselect_savepfp(f):
    return make_midf_mcnu(
        f,
        applyPreselection=False,
        applyBarycenterCut=False,
        savePfp=True,
        include_weights=False,
        pfpTopology="none",
    )


def make_midf_mcnu_preselect_savepfp_wgt(f):
    return make_midf_mcnu(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        include_weights=True,
        pfpTopology="none",
    )


def make_midf_mcnu_nopreselect_savepfp_wgt(f):
    return make_midf_mcnu(
        f,
        applyPreselection=False,
        applyBarycenterCut=False,
        savePfp=True,
        include_weights=True,
        pfpTopology="none",
    )


def make_midf_mcnu_preselect_1shw_savepfp(f):
    return make_midf_mcnu(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        include_weights=False,
        pfpTopology="1shw",
        showerTrackScoreCut=0.5,
    )


def make_midf_mcnu_preselect_2shw_savepfp(f):
    return make_midf_mcnu(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        include_weights=False,
        pfpTopology="2shw",
        showerTrackScoreCut=0.5,
    )


def make_midf_mcnu_preselect_1shw_savepfp_wgt(f):
    return make_midf_mcnu(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        include_weights=True,
        pfpTopology="1shw",
        showerTrackScoreCut=0.5,
    )


def make_midf_mcnu_preselect_2shw_savepfp_wgt(f):
    return make_midf_mcnu(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        include_weights=True,
        pfpTopology="2shw",
        showerTrackScoreCut=0.5,
    )


# ============================================================
# DATA VERSION
# ============================================================

def make_midf_data(
    f,
    applyPreselection=True,
    applyBarycenterCut=True,
    savePfp=True,
    nuScoreCut=0.5,
    barycenterScoreCut=0.02,
    pfpTopology="none",
    showerTrackScoreCut=0.5,
):
    """
    Model-independent dataframe for data.
    Truth columns are removed.
    Frame and timing information are merged.
    """

    slcdf = make_midf(
        f,
        applyPreselection=applyPreselection,
        applyBarycenterCut=applyBarycenterCut,
        savePfp=savePfp,
        nuScoreCut=nuScoreCut,
        barycenterScoreCut=barycenterScoreCut,
        pfpTopology=pfpTopology,
        showerTrackScoreCut=showerTrackScoreCut,
    )

    if "tmatch" in slcdf.columns.get_level_values(1):
        slcdf = slcdf.drop("tmatch", axis=1, level=1)

    if "truth" in slcdf.columns.get_level_values(2):
        slcdf = slcdf.drop("truth", axis=1, level=2)

    framedf = make_framedf(f)
    timingdf = make_timingdf(f)

    ftdf = multicol_merge(
        framedf,
        timingdf,
        left_index=True,
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    df = multicol_merge(
        slcdf.reset_index(),
        ftdf.reset_index(),
        left_on=[("entry", "", "", "", "", "")],
        right_on=[("entry", "", "", "", "", "")],
        how="left",
    )

    df = df.set_index(slcdf.index.names, verify_integrity=True)

    return df


def make_midf_data_preselect_savepfp(f):
    return make_midf_data(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        pfpTopology="none",
    )


def make_midf_data_nopreselect_savepfp(f):
    return make_midf_data(
        f,
        applyPreselection=False,
        applyBarycenterCut=False,
        savePfp=True,
        pfpTopology="none",
    )


def make_midf_data_preselect_1shw_savepfp(f):
    return make_midf_data(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        pfpTopology="1shw",
        showerTrackScoreCut=0.5,
    )


def make_midf_data_preselect_2shw_savepfp(f):
    return make_midf_data(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        pfpTopology="2shw",
        showerTrackScoreCut=0.5,
    )