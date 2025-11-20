#!/bin/bash
#SBATCH --job-name=stage6_sync
#SBATCH --output=/home/mlut/PsyTSLM/.garbage/stage6.out
#SBATCH --error=/home/mlut/PsyTSLM/.garbage/stage6.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40gb:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --partition=acltr
#SBATCH --nodelist=cn13

# cn13: Using 2x A100 40GB (2 GPUs available, 2 already in use by another job)
# Using torchrun instead of srun - handles distributed training setup automatically

module load Anaconda3
source activate sync-opentslm

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: 2 (2 A100 40GB available on cn13)"
echo "Using torchrun for automatic port selection"
echo "Start Time: $(date)"
echo "=========================================="


# RECOMMENDED: torchrun handles all distributed setup automatically
echo "Running Stage 6 on 2 A100s with torchrun..."
torchrun \
  --nproc_per_node=2 \
  --nnodes=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  curriculum_learning.py \
  --model OpenTSLMFlamingo \
  --llm_id meta-llama/Llama-3.2-1B \
  --stages stage6_synchrony_cot \
  --batch_size 4 \
  --gradient_checkpointing

# Effective batch size: 2 GPUs × 4 batch × 2 accum = 16

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
