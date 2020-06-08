# CS 230 Project

This is the repository for the CS230 Project

## Workflow

1. Feature Vector Processing (`feature-vec-processing.ipynb`)
2. Data Merging (`data-merging.ipynb`)
3. Data Exploration (`data-exploration-2.ipynb`)
4. Data Upsampling (`data-upsampling.ipynb`)
5. Baseline Models (`baseline-models-upsampled.ipynb`)
6. Hyperparameter Tuning (`hyperparameter-search-upsampled.py`)
7. Final Model Tuning and Evaluation (`final-models.ipynb`)

## File Descriptions
The following is a description of main scripts and files:

### Exploration
- **data-exploration:** Data exploration for Milestone 0/1
- **data-exploration-2:** Additional exploration for Milestone 2

### Data Processing
- **data-merging:** Pipeline for merging initial dataset with FEATURE vector data
- **feature-vec-processing:** Pipeline for processing FEATURE vector files into CSV
- **data-upsampling**: Pipeline for upsampling the data from the merged dataset

### Model Training and Evaluation

#### Baselines
- **logreg-models:** Logistic Regression models on final merged dataset (Milestone 2)
- **baseline-models**: 2-layer neural network baselines using *original merged dataset*
- **baseline-models-upsampled**: 2-layer neural network baselines using *upsampled merged dataset*

#### Hyperparameter Tuning
- **hyperparameter-search**: Hyperpameter search script using *original merged dataset*
- **hyperparameter-search-upsampled**: Hyperpameter search script using *upsampled merged dataset*
- **final-models**: Final model training and evaluation following hyperparameter tuning


### Archived Files
- **feature-vec-processing (OLD):** Original FETURE vector processing w/errors, kept for archived reference
- **logistic-reg-test:** Early logistic regression training using merged data that had errors. Kept for archived reference

## Protein -> PDB Maps
Reference PDB information
**TEM-1 (P62593):** 1XPB (Chain A)  
**Kka2 (P00552):** 1ND4 (Chain A) - Dimer, so exclude Chain B  
**Uba1 (P0CG63):** 3CMM (Chain A) - Dimer in complex, so exclude B-D  
**PSD95pdz3 (P31016):** 1BE9 (Chain A)  
**Pab1 (P04147):** 1CVJ  (Chain A)   
**hsp90 (P02829):** 2CG9 (Chain A)  
