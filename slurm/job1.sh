#!/bin/bash
#SBATCH --job-name=data-syrup-job1
#SBATCH --partition=gpu
#SBATCH --output="J-%x.out"
#SBATCH --error="J-%x.out"
#SBATCH --gres=gpu:1

echo "SLURM WORKLOAD START: $(date)"
start=$(date +%s)

nvidia-smi

module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

python src/e2e.py configs/job1.json

deactivate
end=$(date +%s)
diff=$((end - start))

echo "TIME TAKEN: $(date -ud "@$diff" +'%T')"
echo "SLURM WORKLOAD FINISH: $(date)"
