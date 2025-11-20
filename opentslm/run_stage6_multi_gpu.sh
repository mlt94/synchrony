#!/bin/bash
#SBATCH --job-name=stage6_sync
#SBATCH --output=/home/mlut/PsyTSLM/.garbage/stage6.out
#SBATCH --error=/home/mlut/PsyTSLM/.garbage/stage6.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100_40gb:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --partition=acltr
#SBATCH --nodelist=cn13

# cn13: Using 2x A100 40GB (2 GPUs available, 2 already in use by another job)
# Resources available: 8 CPUs free (32 total - 24 allocated), ~45GB RAM free

module load Anaconda3
source activate sync-opentslm

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29501  # Changed from 29500 (occupied by other job)
export WORLD_SIZE=2  # Only 2 GPUs available
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand since cn13 only has 1 Gbps Eth
export NCCL_P2P_DISABLE=0  # Enable P2P for faster GPU communication

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: 2 (2 A100 40GB available on cn13)"
echo "Master Port: 29501 (avoiding conflict with other job)"
echo "Start Time: $(date)"
echo "=========================================="


# Option 1: RECOMMENDED - 2 GPUs with gradient accumulation to match 4-GPU effective batch
echo "Running Stage 6 on 2 A100s with gradient accumulation..."
srun python curriculum_learning.py \
  --model OpenTSLMFlamingo \
  --llm_id meta-llama/Llama-3.2-1B \
  --stages stage6_synchrony_cot \
  --batch_size 4 \
  --gradient_checkpointing

# Effective batch size: 2 GPUs × 4 batch × 2 accum = 16
# (vs 4 GPUs × 4 batch × 1 accum = 16 with all GPUs)

# Effective batch size: 2 GPUs × 4 batch × 2 accum = 16
# (vs 4 GPUs × 4 batch × 1 accum = 16 with all GPUs)

# Option 2: Conservative - No mixed precision if you encounter issues
# echo "Running Stage 6 on 2 A100s (conservative mode)..."
# srun python curriculum_learning.py \
#   --model OpenTSLMFlamingo \
#   --llm_id meta-llama/Llama-3.2-1B \
#   --stages stage6_synchrony_cot \
#   --batch_size 4 \
#   --gradient_accumulation_steps 2 \
#   --gradient_checkpointing

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
