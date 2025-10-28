#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=submit
#SBATCH --partition=GPUQ
#SBATCH --time=00:03:00

module purge
module load Anaconda3/2023.09-0
source ~/.bashrc
conda activate dream_env

mkdir -p logs
"""
combos=(
"0"
"1"
"2"
"3"
"4"
"5"
"0 1"
"0 2"
"0 3"
"0 4"
"0 5"
"1 2"
"1 3"
"1 4"
"1 5"
"2 3"
"2 4"
"2 5"
"3 4"
"3 5"
"4 5"
"0 1 2"
"0 1 3"
"0 1 4"
"0 1 5"
"0 2 3"
"0 2 4"
"0 2 5"
"0 3 4"
"0 3 5"
"0 4 5"
"1 2 3"
"1 2 4"
"1 2 5"
"1 3 4"
"1 3 5"
"1 4 5"
"2 3 4"
"2 3 5"
"2 4 5"
"3 4 5"
"0 1 2 3"
"0 1 2 4"
"0 1 2 5"
"0 1 3 4"
"0 1 3 5"
"0 1 4 5"
"0 2 3 4"
"0 2 3 5"
"0 2 4 5"
"0 3 4 5"
"1 2 3 4"
"1 2 3 5"
"1 2 4 5"
"1 3 4 5"
"2 3 4 5"
"0 1 2 3 4"
"0 1 2 3 5"
"0 1 2 4 5"
"0 1 3 4 5"
"0 2 3 4 5"
"1 2 3 4 5"
"0 1 2 3 4 5"
)
"""
combos=(
"0 1 2"
)
for combo in "${combos[@]}"; do
  name="imf_${combo// /_}"
  sbatch <<EOF
#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=${name}
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=600G
#SBATCH --time=200:00:00
#SBATCH --output=logs/${name}.out
#SBATCH --error=logs/${name}.err

module purge
module load Anaconda3/2023.09-0
source ~/.bashrc
conda activate dream_env

python src/main_imf.py --imfs ${combo}
EOF
done
