#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=segmented_2s_epoch_256hz_csp_batch
#SBATCH --time=72:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --output=logs/segmented_2s_epoch_256hz_csp_batch_out.txt
#SBATCH --error=logs/segmented_2s_epoch_256hz_csp_batch_err.txt

module purge
module load Anaconda3/2023.09-0
source ~/.bashrc
conda activate dream_env

subjects=(s01 s02 s03 s04 s05 s06 s07 s08 s09 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s30 s31 s32 s33 s34 s35 s36)

for subj in "${subjects[@]}"; do
sbatch <<EOF
#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=csp_subj_${subj}
#SBATCH --time=72:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=120G
#SBATCH --output=logs/csp_segmented_2s_epoch_256hz_subj_${subj}_out.txt
#SBATCH --error=logs/csp_segmented_2s_epoch_256hz_subj_${subj}_err.txt

module purge
module load Anaconda3/2023.09-0
source ~/.bashrc
conda activate dream_env

python src/main_csp_single.py --subject_id ${subj}
EOF
done
