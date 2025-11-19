#!/bin/bash
#SBATCH --job-name=stage6_sync
#SBATCH --output=home/mlut/PsyTSLM/.garbage/stage6.out
#SBATCH --error=home/mlut/PsyTSLM/.garbage/stage6.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --partition=acltr
#SBATCH --nodelist=cn13

# cn13: 4x Nvidia A100 40GiB - Best option for multi-GPU training

module load Anaconda3
source activate sync-opentslm

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=4
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand since cn13 only has 1 Gbps Eth
export NCCL_P2P_DISABLE=0  # Enable P2P for faster GPU communication

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_NTASKS_PER_NODE"
echo "Start Time: $(date)"
echo "=========================================="


# Option 1: RECOMMENDED - Mixed precision + gradient accumulation for speed
echo "Running Stage 6 with ALL optimizations (FASTEST)..."
srun python curriculum_learning.py \
  --model OpenTSLMFlamingo \
  --llm_id meta-llama/Llama-3.2-1B \
  --stages stage6_synchrony_cot \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --mixed_precision \
  --gradient_checkpointing

# Option 2: Conservative - Standard multi-GPU without mixed precision
# echo "Running Stage 6 with standard multi-GPU..."
# srun python curriculum_learning.py \
#   --model OpenTSLMFlamingo \
#   --llm_id meta-llama/Llama-3.2-1B \
#   --stages stage6_synchrony_cot \
#   --batch_size 4 \
#   --gradient_accumulation_steps 2

# Option 3: Maximum batch size with mixed precision
# echo "Running Stage 6 with maximum batch size..."
# srun python curriculum_learning.py \
#   --model OpenTSLMFlamingo \
#   --llm_id meta-llama/Llama-3.2-1B \
#   --stages stage6_synchrony_cot \
#   --batch_size 8 \
#   --gradient_accumulation_steps 4 \
#   --mixed_precision

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
