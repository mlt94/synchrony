#!/bin/bash
#SBATCH --job-name=baseline_gemma7b
#SBATCH --output=/home/mlut/synchrony/.garbage/baseline.out
#SBATCH --error=/home/mlut/synchrony/.garbage/baseline.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=22:00:00
#SBATCH --partition=acltr
#SBATCH --nodelist=cn7

# Single A100 80GB is sufficient for Gemma 7B inference
# This baseline evaluates Gemma's ability to predict empathy categories
# from text descriptions (without raw time series data)

module load Anaconda3
source activate sync-opentslm

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPU: A100 80GB"
echo "Start Time: $(date)"
echo "=========================================="


# Run baseline evaluation
python baseline.py \
  --data_model /path/to/data_model.yaml \
  --config ../config_opentslm.yaml \
  --answer_dir /path/to/answer/directory \
  --output_dir ./baseline_results \
  --model_name google/gemma-7b-it \
  --eval_split all \
  --save_predictions

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
