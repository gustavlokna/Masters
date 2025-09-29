#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=memd_job
#SBATCH --time=30:00:00

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --nodes=1
#SBATCH --output=memd_out.txt         # ✅ Output file
#SBATCH --error=memd_err.txt          # ✅ Error file

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

# ✅ Activate your conda env
source ~/.bashrc
conda activate dream_env

python src/main.py --memd
