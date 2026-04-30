# Model-Independent Analysis Framework (SBND)

This module provides a model-independent dataframe production pipeline for SBND analyses using CAF files and the `cafpyana` framework.

The goal is to build flexible datasets for physics studies without relying on signal-specific assumptions.

---

##  Features

- Model-independent preselection
- Optional PFP topology selection (1-shower / 2-shower)
- Reco + matched MC truth merging
- Systematic weights support (BNB, GENIE, G4)
- Data and MC compatible workflows
- Optimized weight computation (only for selected slices)

---

## Directory Structure

analysis_village/model_independent/

├── configs/
│   ├── mi_data_nopreselect_savepfp.py
│   ├── mi_data_preselect_savepfp.py
│   ├── mi_mcnu_nopreselect_savepfp.py
│   ├── mi_mcnu_preselect_savepfp.py
│   ├── mi_mcnu_wgt_preselect_1shw_savepfp.py
│   └── mi_mcnu_wgt_preselect_2shw_savepfp.py
│
└── makedf/
    └── make_midf.py

---

##  Usage

Run dataframe production with:

python run_df_maker.py \
  -c analysis_village/model_independent/configs/<CONFIG>.py \
  -l <input_file_list>.list \
  -o <output_path>

Example:

python run_df_maker.py \
  -c analysis_village/model_independent/configs/mi_mcnu_wgt_preselect_1shw_savepfp.py \
  -l my_files.list \
  -o output_dir/

---

## Configuration Options

### Preselection

Applied cuts:

- slc.is_clear_cosmic == 0
- slc.nu_score > 0.5
- Fiducial volume cut
- Optional barycenter cut:
  - slc.barycenterFM.score > 0.02

---

### PFP Topology Selection

Controlled via:

pfpTopology = "none" | "1shw" | "2shw"

Rules:

- PFPs with trackScore == 0 are removed
- Showers defined by:
  pfp.trackScore < 0.5

---

### Systematic Weights

Supported:

- BNB (flux)
- GENIE (cross-section)
- G4 (hadron reinteraction)

 Important implementation detail:

BNB and GENIE weights are computed on the full MC sample, then filtered to selected slices:

full MC → compute weights → filter to selected neutrinos

This avoids:

- shape mismatches
- broadcasting errors
- missing universes

---

##  Output

The final dataframe includes:

### Reco information

- Slice-level variables
- PFP-level variables (if savePfp=True)

### Truth (MC only)

- Neutrino truth information
- Matching via:
  slc.tmatch.idx

### Weights (optional)

- Multisim universes for systematics

---

## Available Configurations

- mi_data_preselect_savepfp → Data with preselection
- mi_data_nopreselect_savepfp → Data without cuts
- mi_mcnu_preselect_savepfp → MC reco + truth (no weights)
- mi_mcnu_nopreselect_savepfp → MC without cuts
- mi_mcnu_wgt_preselect_1shw_savepfp → MC with weights (1 shower)
- mi_mcnu_wgt_preselect_2shw_savepfp → MC with weights (2 showers)


---

##  Author

Gaetano Fricano  
PhD student, University of Palermo  
SBND Collaboration
