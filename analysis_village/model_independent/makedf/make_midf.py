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
    top_levels = list(zip(*list(mcdf.columns)))[0]

    if "mu" in top_levels:
        mcdf = mcdf.drop("mu", axis=1, level=0)
    if "p" in top_levels:
        mcdf = mcdf.drop("p", axis=1, level=0)
    if "cpi" in top_levels:
        mcdf = mcdf.drop("cpi", axis=1, level=0)

    return mcdf


def _make_full_nuind(mcdf_full):
    return pd.Series(
        mcdf_full.index.get_level_values(1).to_numpy(dtype=int),
        index=mcdf_full.index,
        name="ind",
    )


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

    # Remove invalid PFPs only, do not reject the slice because of them
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
    det = loadbranches(f["recTree"], ["rec.hdr.det"]).rec.hdr.det

    if 1 == det.unique()[0]:
        DETECTOR = "SBND"
    else:
        DETECTOR = "ICARUS"

    pfpdf = make_pfpdf(f)

    slcdf = loadbranches(
        f["recTree"],
        slcbranches + barycenterFMbranches + correctedflashbranches
    )
    slcdf = slcdf.rec

    if "pfochar" in pfpdf.columns.get_level_values(1):
        pfpdf = pfpdf.drop("pfochar", axis=1, level=1)

    # --------------------------------------------------------
    # Save all PFPs
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
    # Save only primary / secondary showers
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

    slcdf = make_midf_preselection(
        slcdf,
        det=DETECTOR,
        applyPreselection=applyPreselection,
        applyBarycenterCut=applyBarycenterCut,
        nuScoreCut=nuScoreCut,
        barycenterScoreCut=barycenterScoreCut,
    )

    if savePfp:
        slcdf = apply_mi_pfp_topology_selection(
            slcdf,
            pfpTopology=pfpTopology,
            showerTrackScoreCut=showerTrackScoreCut,
        )

    return slcdf


# ============================================================
# MC NEUTRINO TRUTH DF
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
    """
    Build MC neutrino truth dataframe and add weights.

    BNB and GENIE are built on the full neutrino sample and then filtered,
    because getsyst expects the full neutrino index.
    G4 is kept on the selected sample.
    """

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

    full_nuind = _make_full_nuind(mcdf_full)

    if "bnb" in wgt_types:
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
# RECO + MATCHED MC NEUTRINO TRUTH
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
# RECO + MATCHED MC MEVPRTL TRUTH
# ============================================================

def make_midf_mevprtl(
    f,
    include_weights=False,
    multisim_nuniv=100,
    slim=True,
    applyPreselection=True,
    applyBarycenterCut=True,
    savePfp=True,
    nuScoreCut=0.5,
    barycenterScoreCut=0.02,
    pfpTopology="none",
    showerTrackScoreCut=0.5,
):
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

    prtldf = make_mevprtldf(
        f,
        include_weights=include_weights,
        multisim_nuniv=multisim_nuniv,
        slim=slim,
    )

    prtldf.columns = pd.MultiIndex.from_tuples(
        [tuple(["slc", "prtl"] + list(c)) for c in prtldf.columns]
    )

    df = multicol_merge(
        slcdf.reset_index(),
        prtldf.reset_index(),
        left_on=[
            ("entry", "", "", "", "", ""),
            ("slc", "tmatch", "idx", "", "", ""),
        ],
        right_on=[
            ("entry", "", "", "", "", ""),
            ("rec.mc.prtl..index", "", ""),
        ],
        how="left",
    )

    df = df.set_index(slcdf.index.names, verify_integrity=True)

    return df


# ============================================================
# MC NEUTRINO WRAPPERS
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
# MC MEVPRTL WRAPPERS
# ============================================================

def make_midf_mevprtl_preselect_savepfp(f):
    return make_midf_mevprtl(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        include_weights=False,
        pfpTopology="none",
    )


def make_midf_mevprtl_nopreselect_savepfp(f):
    return make_midf_mevprtl(
        f,
        applyPreselection=False,
        applyBarycenterCut=False,
        savePfp=True,
        include_weights=False,
        pfpTopology="none",
    )


def make_midf_mevprtl_preselect_1shw_savepfp(f):
    return make_midf_mevprtl(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        include_weights=False,
        pfpTopology="1shw",
        showerTrackScoreCut=0.5,
    )


def make_midf_mevprtl_preselect_2shw_savepfp(f):
    return make_midf_mevprtl(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        include_weights=False,
        pfpTopology="2shw",
        showerTrackScoreCut=0.5,
    )


def make_midf_mevprtl_preselect_savepfp_wgt(f):
    return make_midf_mevprtl(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        include_weights=True,
        pfpTopology="none",
    )


def make_midf_mevprtl_nopreselect_savepfp_wgt(f):
    return make_midf_mevprtl(
        f,
        applyPreselection=False,
        applyBarycenterCut=False,
        savePfp=True,
        include_weights=True,
        pfpTopology="none",
    )


def make_midf_mevprtl_preselect_1shw_savepfp_wgt(f):
    return make_midf_mevprtl(
        f,
        applyPreselection=True,
        applyBarycenterCut=True,
        savePfp=True,
        include_weights=True,
        pfpTopology="1shw",
        showerTrackScoreCut=0.5,
    )


def make_midf_mevprtl_preselect_2shw_savepfp_wgt(f):
    return make_midf_mevprtl(
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