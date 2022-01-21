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

### 10-fold CV
In order to run the 10-fold stratified cross-validation on the cluster of the Freie Universität Berlin, use the bash script below called `cytotox_10_fold_cross_validation.sh` (by adapting your email):

```bash
#!/bin/bash
#SBATCH --job-name=cytotox_model_10_fold
#SBATCH --mail-user=XXX
#SBATCH --mail-type=end
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=8:00:00
#SBATCH --qos=standard

declare -a combinations
index=0

K=10 # K-fold cross-validation

for fold in {0..9}
do
	combinations[$index]="$fold"
	index=$((index + 1))
done

parameters=(${combinations[${SLURM_ARRAY_TASK_ID}]})
fold=${parameters[0]}

python3 scripts/train_tox.py data/cytotoxicity/10_fold_stratified_cross_validation/cytotox_train${fold}-$K.csv \
data/cytotoxicity/10_fold_stratified_cross_validation/cytotox_test${fold}-$K.csv data/cytotoxicity/cytotox.smi data/smiles_language_tox21.pkl \
models${fold}-$K params/mca.json test --embedding_path data/smiles_vae_embeddings.pkl
```

And execute the bash script by running the following:

```console
sbatch --array=0-9 cytotox_10_fold_cross_validation.sh
```