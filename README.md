# RL_NTL - VERL Reinforcement Learning for Large Language Models

This repository contains the VERL (Versatile Efficient Reinforcement Learning) framework for training LLMs with PPO on mathematical reasoning tasks like GSM8K.

## Project Structure

```
RL_NTL/
├── verl/                    # Main VERL framework
│   ├── trainer/            # Training scripts and configs
│   ├── models/             # Model implementations (LLaMA, Qwen2)
│   ├── workers/            # Distributed workers for actor, critic, reward
│   ├── utils/              # Utilities and reward functions
│   └── tools/              # GSM8K and other task-specific tools
├── data/                   # Dataset directory (created by prepare script)
├── setup.py                # Package installation
├── prepare_gsm8k_data.sh   # Data preparation script
├── train_gsm8k_qwen_ppo.sh # Main training script
└── custom_gsm8k_reward.py  # Example custom reward function
```

## Setup

### 1. Environment Setup

```bash
# IMPORTANT: Always source the NTL environment first
source /home/yeopjin/orcd/pool/init_NTL.sh

# Install the package (if not already installed)
pip install -e .
```

### 2. Data Preparation

```bash
# Make sure environment is sourced first!
source /home/yeopjin/orcd/pool/init_NTL.sh

# Download and prepare GSM8K dataset
./prepare_gsm8k_data.sh
```

This will:
- Download the GSM8K dataset
- Process it into train/test parquet files
- Save to `./data/` directory

### 3. Training

#### Basic Training (4 GPUs)
```bash
# Always source environment first
source /home/yeopjin/orcd/pool/init_NTL.sh

# Run training
./train_gsm8k_qwen_ppo.sh
```

This will:
- Set `WANDB_PROJECT='RL-NTL'`
- Generate experiment name with date: `gsm8k-ppo-qwen2.5-0.5b-{date}`
- Log output to both console and `{EXPERIMENT_NAME}.log`
- Save checkpoints to `checkpoints/RL-NTL/{EXPERIMENT_NAME}/`

#### Custom Configuration
```bash
# Override experiment name
export EXPERIMENT_NAME="my-custom-experiment"
./train_gsm8k_qwen_ppo.sh

# Change model
export BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
./train_gsm8k_qwen_ppo.sh
```

## Configuration

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.train_batch_size` | 1024 | Training batch size |
| `data.max_prompt_length` | 1024 | Maximum prompt length |
| `data.max_response_length` | 512 | Maximum response length |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 | Actor learning rate |
| `critic.optim.lr` | 1e-5 | Critic learning rate |
| `trainer.n_gpus_per_node` | 4 | Number of GPUs |
| `trainer.total_epochs` | 15 | Training epochs |

## Reward System

### Default GSM8K Reward

The default reward function for GSM8K:
- **Correct answer**: 1.0 reward
- **Wrong answer**: 0.0 reward
- **Format**: Expects answers in `#### NUMBER` format

### Custom Reward Functions

You can create custom reward functions to modify scoring behavior.

#### 1. Create a Python file with your reward function:

```python
# custom_reward.py
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Custom scoring function.
    
    Args:
        data_source: Dataset identifier (e.g., "openai/gsm8k")
        solution_str: Model's response
        ground_truth: Correct answer
        extra_info: Additional metadata
    
    Returns:
        float: Reward score (typically 0.0 to 1.0)
    """
    # Your custom scoring logic here
    # Example: Give partial credit for showing work
    
    import re
    
    # Extract answer
    match = re.search(r"#### ([\d.,-]+)", solution_str)
    if not match:
        return 0.0
    
    answer = match.group(1).replace(",", "")
    
    # Full credit for correct answer
    if answer == ground_truth:
        return 1.0
    
    # Partial credit for showing steps
    if "step" in solution_str.lower() or "=" in solution_str:
        return 0.3
    
    return 0.0
```

#### 2. Configure training to use custom reward:

In your training script, add:
```bash
custom_reward_function.path=/path/to/custom_reward.py \
custom_reward_function.name=compute_score \
```

### Example Custom Reward Functions

The repository includes `custom_gsm8k_reward.py` with examples:

1. **Basic custom scoring** - Configurable correct/format scores
2. **Step-based scoring** - Rewards for showing intermediate steps
3. **Lenient scoring** - Accepts multiple answer formats

### Advanced Reward Managers

VERL supports different reward managers:
- `naive`: Simple reward computation (default)
- `prime`: Parallel reward computation for efficiency
- `batch`: Batched reward processing
- `format`: Format-aware scoring for structured outputs

Configure with:
```bash
reward_model.reward_manager=prime \
```

## Model Support

Currently supports:
- Qwen2.5 models (0.5B, 1.5B, 7B, etc.)
- LLaMA models
- Custom models via HuggingFace

## Distributed Training

The framework uses:
- **FSDP** (Fully Sharded Data Parallel) for model parallelism
- **vLLM** for efficient inference during rollout
- **Ray** for distributed orchestration

## Monitoring

### Console Logging
By default, training progress is logged to the console.

### Weights & Biases (wandb) Integration

VERL supports wandb for experiment tracking and visualization.

#### Setup wandb:
```bash
# Login to wandb (one-time setup)
wandb login

# Or set API key
export WANDB_API_KEY="your_api_key_here"
```

#### Enable wandb in training:
```bash
trainer.logger="['console', 'wandb']" \
```

#### Configure wandb (optional):
```bash
# Set environment variables before training
export WANDB_PROJECT="verl-gsm8k"
export WANDB_ENTITY="your_team"
export WANDB_RUN_NAME="experiment_1"
export WANDB_TAGS="gsm8k,qwen,ppo"
export WANDB_NOTES="Testing new reward function"

# Or run in offline mode
export WANDB_MODE=offline
```

#### What's logged to wandb:
- Training metrics (loss, rewards, KL divergence)
- Validation metrics and accuracies
- Learning rates and gradient norms
- Sample generations (with `trainer.log_val_generations=5`)
- System metrics (GPU usage, memory)
- Hyperparameters

#### Example with wandb:
```bash
./train_gsm8k_wandb.sh
```

### Checkpoints
Model checkpoints are saved to: `checkpoints/{project_name}/{experiment_name}/`

## Tips

1. **Memory Management**: 
   - Adjust `actor_rollout_ref.rollout.gpu_memory_utilization` (default: 0.4)
   - Enable parameter offloading for large models

2. **Batch Sizes**:
   - Start with smaller batches if OOM
   - `ppo_micro_batch_size`: Per-GPU gradient accumulation

3. **Custom Datasets**:
   - Modify data preprocessing in `verl/examples/data_preprocess/`
   - Ensure your data has `prompt` and `ground_truth` fields

## Troubleshooting

- **CUDA OOM**: Reduce batch sizes or enable offloading
- **Slow training**: Check tensor parallel size matches GPU count
- **Poor rewards**: Verify reward function and data format
- **Connection errors**: Ensure Ray cluster is properly initialized

## Citation

If you use VERL in your research, please cite:
```bibtex
@misc{verl2024,
  title={VERL: Versatile Efficient Reinforcement Learning},
  author={Bytedance Ltd.},
  year={2024}
}
```