#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=memd_batch
#SBATCH --time=240:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --output=logs/memd_batch_out.txt
#SBATCH --error=logs/memd_batch_err.txt
#SBATCH --mail-user=gustal@stud.ntnu.no
#SBATCH --mail-type=ALL

#SBATCH --mail-user=gustal@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from: $WORKDIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID" 
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2023.09-0
source ~/.bashrc
conda activate dream_env

# list of subjects
#(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36)
subjects= (1 2) 


for subj in "${subjects[@]}"; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=memd_subj_${subj}
#SBATCH --time=240:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem= 240G

#SBATCH --nodes=1
#SBATCH --output=logs/memd_subj_${subj}_out.txt
#SBATCH --error=logs/memd_subj_${subj}_err.txt

module purge
module load Anaconda3/2023.09-0
source ~/.bashrc
conda activate dream_env

python src/main_memd_single.py --subject_id ${subj}
EOF
done
