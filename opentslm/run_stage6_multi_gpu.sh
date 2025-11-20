#!/bin/bash
#SBATCH --job-name=stage6_sync
#SBATCH --output=/home/mlut/PsyTSLM/.garbage/stage6.out
#SBATCH --error=/home/mlut/PsyTSLM/.garbage/stage6.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40gb:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=4-00:00:00
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
echo "Number of GPUs: 4"
echo "Using torchrun for automatic port selection"
echo "Start Time: $(date)"
echo "=========================================="


# RECOMMENDED: torchrun handles all distributed setup automatically
echo "Running Stage 6 on 4 A100s with torchrun..."

# Try torchrun as python module first (most reliable for PyTorch 2.x)
if python -c "import torch.distributed.run" &> /dev/null; then
    echo "Using python -m torch.distributed.run (PyTorch 2.x torchrun)"
    python -m torch.distributed.run \
      --nproc_per_node=2 \
      --nnodes=1 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=localhost:0 \
      curriculum_learning.py \
      --model OpenTSLMFlamingo \
      --llm_id meta-llama/Llama-3.2-1B \
      --stages stage6_synchrony_cot \
      --batch_size 2 \
      --gradient_checkpointing
# Try torchrun command (if in PATH)
elif command -v torchrun &> /dev/null; then
    echo "Using torchrun command"
    torchrun \
      --nproc_per_node=2 \
      --nnodes=1 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=localhost:0 \
      curriculum_learning.py \
      --model OpenTSLMFlamingo \
      --llm_id meta-llama/Llama-3.2-1B \
      --stages stage6_synchrony_cot \
      --batch_size 2 \
      --gradient_checkpointing
else
    # Fallback to old torch.distributed.launch
    echo "Using torch.distributed.launch (fallback)"
    python -m torch.distributed.launch \
      --use-env \
      --nproc_per_node=2 \
      --nnodes=1 \
      --master_addr=localhost \
      --master_port=0 \
      curriculum_learning.py \
      --model OpenTSLMFlamingo \
      --llm_id meta-llama/Llama-3.2-1B \
      --stages stage6_synchrony_cot \
      --batch_size 2 \
      --gradient_checkpointing
fi

# Effective batch size: 2 GPUs × 4 batch × 2 accum = 16

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
