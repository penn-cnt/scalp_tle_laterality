TLE laterality project
This is the codebase for the project using interictal spike scalp EEG data to lateralize left vs. right/bilateral TLE or right vs. left/bilateral TLE. Here we describe steps to begin with a dataset containing spike rates and calculate spike asymmetry index (SAI). Additionally, code to perform univariate analysis across spike rates and SAI and initializing machine learning models to predict left vs. rest and right vs. rest is included.
To run the analysis, follow these steps:
1. Download the codebase from: https://github.com/penn-cnt/scalp_tle_laterality.
2. Download the datasets from: https://github.com/penn-cnt/scalp_tle_laterality/Dataset.

Explaining codebase:
* Dataset folder contains intermediate datasets and features that can be used to replicate study. It has spikes in left and right hemisphere (from our LOHO method) in all, wake, and sleep states (from sleep staging algorithm YASA) separately.
* Univariate_Analysis folder contains spike rates and SAI analysis in model development cohorts. Code here replicates figure 2B and 2C (complete run in <1 min).
* Machine_Learning folder contains python script for creating our logistic regression model and post-hoc analysis. Code here replicates figure 3C and 3D.
* Outcome folder contains the surgical outcome analysis (Fig 4B and 4D).
* Visualization folder stores all outputs of the analysis.
* tools folder contains some of the basic functions that are used in the analysis.

Web-based calculator is: https://penn-cnt.github.io/scalp_lateralization/