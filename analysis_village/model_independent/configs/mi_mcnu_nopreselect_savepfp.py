from analysis_village.model_independent.makedf.make_midf import *

DFS = [
    make_midf_mcnu_nopreselect_savepfp,
    make_hdrdf,
    make_potdf_bnb,
    make_mcnudf,
]

NAMES = [
    "mi_rec",
    "hdr",
    "pot",
    "mcnu",
]