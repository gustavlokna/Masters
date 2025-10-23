#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=psd_batch
#SBATCH --time=1:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --output=logs/psd_batch_out.txt
#SBATCH --error=logs/psd_batch_err.txt
#SBATCH --mail-user=gustal@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID" 
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2023.09-0
source ~/.bashrc
conda activate dream_env

files=(
segmented_10s_epoch_256hz.npz
segmented_15s_epoch_256hz.npz
segmented_1s_epoch_256hz.npz
segmented_20s_epoch_256hz.npz
segmented_2.5s_epoch_256hz_no_bp_filter.npz
segmented_25s_epoch_256hz.npz
segmented_2.5s_epoch_256hz.npz
segmented_2s_epoch_256hz.npz
segmented_30s_epoch_256hz.npz
segmented_3s_epoch_256hz.npz
segmented_5s_epoch_256hz.npz
)

for f in "${files[@]}"; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=psd_${f}
#SBATCH --time=48:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --output=logs/psd_${f}_out.txt
#SBATCH --error=logs/psd_${f}_err.txt

module purge
module load Anaconda3/2023.09-0
source ~/.bashrc
conda activate dream_env

python src/main_psd.py --file ${f}
EOF
done
