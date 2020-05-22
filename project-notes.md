# Project Notes

## Protein -> PDB Map

**TEM-1 (P62593):** 1XPB (Chain A)
**Kka2 (P00552):** 1ND4 (Chain A) - Dimer, so exclude Chain B
**Uba1 (P0CG63):** 3CMM (Chain A) - Dimer in complex, so exclude B-D
**PSD95pdz3 (P31016):** 1BE9 (Chain A)
**Pab1 (P04147):** 1CVJ  (Chain A)
**Yap65 (P46937):** 1JMQ (Chain A)
**hsp90 (P02829):** 2CG9 (Chain A)
**GB1 (P06654):** 1PGA (Chain A)

## Milestone 2 Goals
- [x] Generate FEATURE vector encodings for local microenviroment
- [x] Develop pipeline to merge FEATURE vector encodings with original dataset
- [ ] Rerun normal logistic regression with one-hot encodings of original data only
    - Needed to explore impact of high-dimensional representation of local microenvironment
- [ ] Run logistic regression with only FEATURE vector data
    - See how good only capturing the microenvironment would be
- [ ] Run logistic regression with only FEATURE vector data *and* one-hot encodings of WT-Mutant amino acids