#!/bin/bash
#SBATCH --job-name=job2
#SBATCH --partition=gpu
#SBATCH --output="logs/J-%x.out"
#SBATCH --error="logs/J-%x.out"
#SBATCH --gres=gpu:a100_80:1

echo "SLURM WORKLOAD START: $(date)"
start=$(date +%s)

nvidia-smi

#module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

python src/e2e.py configs/no_rag.json

deactivate
end=$(date +%s)
diff=$((end - start))

echo "TIME TAKEN: $(date -ud "@$diff" +'%T')"
echo "SLURM WORKLOAD FINISH: $(date)"
