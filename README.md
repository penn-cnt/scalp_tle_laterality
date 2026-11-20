TLE laterality project
This is the codebase for the project using interictal spike scalp EEG data to lateralize left vs. right/bilateral TLE or right vs. left/bilateral TLE. Here we describe steps to begin with a dataset containing spike rates and calculate spike asymmetry index (SAI). Additionally, code to perform univariate analysis across spike rates and SAI and initializing machine learning models to predict left vs. rest and right vs. rest is included.
To run the analysis, follow these steps:
1. Download the codebase from: https://github.com/penn-cnt/scalp_tle_laterality.
2. Download the datasets from: https://github.com/penn-cnt/scalp_tle_laterality/Dataset.

Explaining codebase:
* Dataset folder contains intermediate datasets and features that can be used to replicate study. It has spikes in left and right hemisphere (from our LOHO method) in all, wake, and sleep states (from sleep staging algorithm YASA) separately.
  * `training_261.csv`: model development cohort (261 patients).
  * `test_104_no_seizures.csv`: test cohort (104 patients). Spike rates and SAI are computed after removing EEG from 30 min before seizure onset to 30 min after seizure offset.
  * `outcome.csv`: 49 patients from both cohorts who had left or right temporal lobe surgery (`Surgery`: Laser Ablation or Resection). `single_lat` is the side of surgery; `engel_stage` and `ilae_stage` are the Engel and ILAE classes at the latest available follow-up (36, 24, 12 or 6 months).
* Univariate_Analysis folder contains spike rates and SAI analysis in model development cohorts. Code here replicates figure 2B and 2C (complete run in <1 min).
* Machine_Learning folder contains python script for creating our logistic regression model and post-hoc analysis. Code here replicates figure 3C and 3D.
* Outcome folder contains the surgical outcome analysis (Fig 4B and 4D).
  * `concordance.py`: trains left vs. rest and right vs. rest SAI models (threshold 0.43) on `training_261.csv`, computes the probability that the model agrees with the side of surgery for operated patients in both cohorts, and compares it between good and poor outcome (Mann-Whitney U, two-sided): Engel I vs. II+ (Fig 4B, p = 0.037) and ILAE 1-2 vs. 3+ (Fig 4D, p = 0.084).
* Visualization folder stores all outputs of the analysis.
* tools folder contains some of the basic functions that are used in the analysis.

Web-based calculator is: https://penn-cnt.github.io/scalp_lateralization/

To set up your environment, first create a conda environment with Python 3.10 and pip:

```bash
conda create -n tle_env python=3.10 pip
conda activate tle_env
```

Then, install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```
