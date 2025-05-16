#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --output="logs/J-%x.out"
#SBATCH --error="logs/J-%x.out"
#SBATCH --gres=gpu:1

echo "SLURM WORKLOAD START: $(date)"
start=$(date +%s)

nvidia-smi

module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

python src/e2e.py configs/rag.json
python src/e2e.py configs/no_rag.json
python src/e2e.py configs/perfect_rag.json

deactivate
end=$(date +%s)
diff=$((end - start))

echo "TIME TAKEN: $(date -ud "@$diff" +'%T')"
echo "SLURM WORKLOAD FINISH: $(date)"
