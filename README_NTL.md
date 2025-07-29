# Number Token Loss (NTL) Integration for RL_NTL

This directory contains a complete integration of Number Token Loss concepts into the VERL PPO training pipeline for enhanced mathematical reasoning.

## What is Number Token Loss?

Number Token Loss (NTL) addresses a fundamental limitation in traditional cross-entropy loss: when predicting numerical tokens, the distance between predicted and actual values should matter. For example, predicting "3" when the answer is "4" should incur less penalty than predicting "9".

## Architecture Overview

### Core Components

1. **`ntl_local/`** - Local NTL implementation
   - `ntl_core.py` - Core NTL loss functions (MSE and Wasserstein variants)
   - `ntl_utils.py` - Utility functions for digit token extraction
   - `__init__.py` - Package initialization

2. **`verl/utils/torch_functional_ntl.py`** - Extended torch functions with NTL
   - Digit-level log probability extraction
   - NTL-augmented loss computation
   - Integration hooks for reward functions

3. **`verl/workers/actor/dp_actor_ntl.py`** - NTL-enabled PPO actor
   - Extends DataParallelPPOActor with digit extraction
   - Captures digit-level information during forward pass
   - Provides NTL metrics for logging

4. **`custom_NTL.py`** - Enhanced reward function
   - Uses digit-level log probabilities for nuanced rewards
   - Multiple scoring strategies (basic, enhanced, fully NTL-aware)
   - Configurable NTL bonus weights

## How It Works

### 1. Digit Extraction During Decoding

During the PPO forward pass, the system:
- Identifies digit tokens (0-9) in the generated sequences
- Extracts log probabilities specifically for digit positions  
- Computes digit-level accuracy and confidence metrics

### 2. NTL Loss Computation

Two NTL variants are implemented:
- **NTL-MSE**: `L = (y_true - E[predicted_digit])^2`
- **NTL-Wasserstein**: Uses optimal transport between digit distributions

### 3. Enhanced Reward Calculation

The reward function now receives:
- Standard text-based correctness score
- Digit-level log probabilities for all digit positions
- NTL loss values and accuracy metrics
- Confidence scores for digit predictions

## Usage

### Training with NTL

**Option 1: NTL with All Digits (Original NTL approach)**
```bash
./train_gsm8k_qwen_ppo_ntl.sh
```

**Option 2: NTL with Final Answer Only (Focused approach)**
```bash
./train_gsm8k_qwen_ppo_ntl_final_answer.sh
```

Both scripts enable:
- `actor_rollout_ref.actor.ntl_enabled=true`
- `actor_rollout_ref.actor.extract_digit_info=true`
- `custom_reward_function.use_ntl_bonus=true`

The difference is the `ntl_bonus_type` parameter:
- `all`: Analyzes all digit tokens in the sequence
- `final_answer`: Focuses only on final answer digits

### Configuration Options

#### Actor Configuration
- `ntl_enabled`: Enable/disable NTL extraction (default: true)
- `ntl_method`: 'mse' or 'wasserstein' (default: 'mse')
- `ntl_weight`: Weight for NTL loss component (default: 0.1)
- `extract_digit_info`: Extract digit-level information (default: true)

#### Reward Function Configuration
- `use_ntl_bonus`: Use NTL information in rewards (default: true)
- `ntl_bonus_type`: 'all' or 'final_answer' (default: 'all')
- `ntl_bonus_weight`: Weight for NTL bonus (default: 0.1)

### Example Configuration

```python
# In your config
actor_rollout_ref:
  actor:
    ntl_enabled: true
    ntl_method: mse
    ntl_weight: 0.1
    extract_digit_info: true

custom_reward_function:
  use_ntl_bonus: true
  ntl_bonus_type: all  # or 'final_answer'
  ntl_bonus_weight: 0.1
```

## Reward Function Strategies

### 1. Basic NTL Enhancement (`compute_score`)
- Standard correctness + configurable NTL bonus
- Two modes: `ntl_bonus_type='all'` or `ntl_bonus_type='final_answer'`
- Safe fallback to original behavior if NTL fails

### 2. NTL All Digits (`compute_ntl_bonus_all`)
- **Original NTL approach**: Analyzes ALL digit tokens in sequence
- Rewards good intermediate calculations AND final answer
- Higher digit accuracy = better reasoning throughout

### 3. NTL Final Answer Only (`compute_ntl_bonus_final_answer`)  
- **Focused approach**: Only analyzes digits in final answer (after "####")
- 40% digit accuracy + 30% confidence + 30% probability quality
- More targeted reward for getting the final answer right

### 4. Fully NTL-Enhanced (`compute_score_ntl_enhanced`)
- 60% correctness + 25% digit accuracy + 15% probability quality
- Most sophisticated digit-level scoring

### 5. Supporting Functions
- `find_final_answer_positions()` - Locates final answer digits in sequence
- `calculate_final_answer_digit_accuracy()` - Accuracy for final answer only
- `calculate_final_answer_confidence()` - Confidence for final answer digits
- `calculate_final_answer_prob_quality()` - Probability quality for final answer

## Key Benefits

1. **Numerical Awareness**: Model learns better digit prediction patterns
2. **Partial Credit**: Rewards for close numerical predictions
3. **Confidence Tracking**: Monitors model confidence in digit predictions
4. **Research Alignment**: Based on published NTL research methodology
5. **Non-Disruptive**: Maintains compatibility with existing pipeline

## Monitoring and Metrics

The NTL integration adds several metrics:
- `ntl/digit_accuracy`: Accuracy of digit token predictions
- `ntl/total_digits`: Number of digit tokens processed
- `ntl/ntl_loss`: NTL loss value
- `ntl/ntl_reward`: NTL-derived reward signal

These appear in your training logs and Wandb dashboard.

## Files Modified/Added

### New Files
- `ntl_local/` (complete package)
- `verl/utils/torch_functional_ntl.py`
- `verl/workers/actor/dp_actor_ntl.py`
- `train_gsm8k_qwen_ppo_ntl.sh`

### Modified Files
- `custom_NTL.py` (enhanced with NTL capabilities)

### Original Files (Unchanged)
- All original VERL components remain intact
- Existing training script continues to work
- Complete backward compatibility maintained

## Research Background

This implementation is based on:
- "Regress, Don't Guess – A Regression-like Loss on Number Tokens for Language Models"
- Paper accepted at ICML 2025
- Original repository: https://github.com/tum-ai/number-token-loss

The key insight: numerical proximity should be reflected in loss computation, enabling more nuanced learning of mathematical reasoning patterns.

## Troubleshooting

### Common Issues

1. **No digit tokens found**: Check tokenizer digit mapping in logs
2. **NTL info missing**: Ensure `extract_digit_info=true` and `ntl_enabled=true`
3. **Memory issues**: NTL adds minimal overhead, but monitor GPU usage
4. **Import errors**: Ensure PYTHONPATH includes the project root

### Debug Mode

Enable detailed logging by setting:
```python
import logging
logging.getLogger('ntl').setLevel(logging.DEBUG)
```

This provides detailed information about digit extraction and NTL computation.