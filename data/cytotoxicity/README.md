# README.md

This README.md explains what lives in the `cytotoxicity` subfolder.

Similar to what exists in the `data` folder, this subfolder should contain:

- `cytotox_train.csv`:
  - The labels and the mol_ids in the train set
- `cytotox_test.csv`:
  - The labels and the mol_ids in the test set
- `cytotox_score.csv`:
  - The labels for all entries in the cytotoxicity data set
- `cytotox.smi`:
  - All SMILES in the cytotoxicity data set along with their mold_id


Moreover,

- `10_fold_stratified_cross_validation/`: A subfolder which contains the train / test split using 10-fold stratified cross-validation, of the form `cytotox_train0-10.csv` and `cytotox_test0-10.csv`.