# Number Token Loss (NTL) Integration for VERL PPO Training

**Date**: July 29, 2025  
**Framework**: VERL PPO with Qwen2.5-0.5B-Instruct  
**Purpose**: Integrate Number Token Loss for improved numerical reasoning

---

## Overview

This implementation integrates Number Token Loss (NTL) into the VERL PPO training pipeline to improve mathematical reasoning capabilities. NTL treats digit prediction as a regression problem rather than discrete classification, providing nuanced feedback for numerical reasoning tasks.

**Key Features:**
- ✅ Two distinct NTL modes: Final Answer and All Tokens
- ✅ Configurable temperature parameter (Tau) for reward scaling
- ✅ Regression-based digit prediction with continuous feedback
- ✅ Multi-digit answer support (e.g., "186" → [1,8,6])
- ✅ Compatible with Qwen2.5 tokenization
- ✅ Comprehensive testing and validation

---

## Two NTL Modes

### **Mode 1: Final Answer (`custom_ntl_final.py`)**

**Reward Formula:**
```python
reward = exp(-ntl_loss / Tau)
```

**Characteristics:**
- **Pure NTL-based reward** - No exact match requirement
- **Continuous learning** - Always provides feedback regardless of correctness
- **Final answer focus** - NTL loss calculated only on final answer digits  
- **True ground truth** - Uses dataset ground truth for final answer evaluation
- **Range**: [0, 1] where 1.0 = perfect digit prediction

**Use Case:** When you want to improve digit-level prediction quality without requiring exact match correctness.

### **Mode 2: All Tokens (`custom_ntl_all.py`)**

**Reward Formula:**
```python
reward = exact_match × exp(-ntl_loss / Tau)
```

**Characteristics:**
- **Hybrid approach** - Combines exact match requirement with NTL quality
- **Zero for wrong answers** - Multiplicative scoring means incorrect answers get no reward
- **Comprehensive coverage** - NTL loss calculated over ALL digit tokens in sequence
- **Generated ground truth** - Uses model's own generated digits as ground truth for intermediate steps
- **Range**: [0, 1] where reward requires both correctness AND good NTL quality

**Use Case:** When you want to maintain exact match accuracy while improving numerical reasoning quality.

---

## Temperature Parameter (Tau)

The temperature parameter Tau controls the sensitivity of the exponential reward function:

```python
# Without temperature (default)
reward = exp(-ntl_loss)

# With temperature
reward = exp(-ntl_loss / Tau)
```

**Effect of Different Tau Values:**

| Tau | Effect | Example (ntl_loss=1.0) |
|-----|--------|------------------------|
| **0.5** | More sensitive | exp(-1.0/0.5) = exp(-2.0) = 0.135 |
| **1.0** | Standard | exp(-1.0/1.0) = exp(-1.0) = 0.368 |
| **2.0** | Less sensitive | exp(-1.0/2.0) = exp(-0.5) = 0.607 |

- **Lower Tau (< 1.0)**: More aggressive - larger penalty for poor NTL quality
- **Higher Tau (> 1.0)**: More forgiving - gentler penalty for poor NTL quality

---

## File Structure

```
RL_NTL/
├── custom_ntl_final.py              # Final answer mode reward function
├── custom_ntl_all.py                # All tokens mode reward function
├── ntl_local/                       # Local NTL implementation
│   ├── __init__.py
│   ├── ntl_core.py                  # Core NTL loss calculation
│   └── ntl_utils.py                 # Utility functions
├── verl/
│   ├── utils/torch_functional_ntl.py # NTL-extended utilities
│   └── workers/actor/dp_actor_ntl.py # NTL-enabled PPO actor
├── train_gsm8k_qwen_ppo_ntl_final_answer.sh  # Final answer training
├── train_gsm8k_qwen_ppo_ntl.sh              # All tokens training
├── test_both_ntl_modes.py           # Comprehensive testing
└── README_NTL.md                    # This file
```

---

## Training Configuration

### **Final Answer Mode Training**

```bash
./train_gsm8k_qwen_ppo_ntl_final_answer.sh

# Key parameters in the script:
custom_reward_function.path=.../custom_ntl_final.py
custom_reward_function.name=compute_score
custom_reward_function.tau=1.0    # Temperature parameter
```

### **All Tokens Mode Training**

```bash
./train_gsm8k_qwen_ppo_ntl.sh

# Key parameters in the script:
custom_reward_function.path=.../custom_ntl_all.py  
custom_reward_function.name=compute_score
custom_reward_function.tau=1.0    # Temperature parameter
```

### **Configurable Parameters**

Both training scripts support these configurable parameters:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `custom_reward_function.tau` | Temperature for exp(-loss/Tau) | 1.0 | 0.1 - 5.0 |

**Example Usage:**
```bash
# More sensitive NTL rewards
custom_reward_function.tau=0.5

# More forgiving NTL rewards  
custom_reward_function.tau=2.0
```

---

## NTL Loss Calculation Methods

We implement two distinct approaches for calculating NTL loss, each with different mathematical formulations and training objectives.

### **Method 1: Negative Log-Likelihood Loss (`custom_ntl_all.py`)**

This method directly applies the negative log-likelihood of the ground truth digits, treating digit prediction as a classification problem with continuous feedback.

**Mathematical Formulation:**

```python
# For each digit position i with ground truth digit d_i:
loss_i = -log(P(d_i | logits_i))

# Where P(d_i | logits_i) is the softmax probability of the true digit
P(d_i | logits_i) = exp(logits_i[d_i]) / Σ(exp(logits_i[j]) for j in 0..9)

# Final NTL loss = mean(losses across all digit positions)
ntl_loss = (1/N) × Σ(loss_i for i in digit_positions)
```

**Implementation:**
```python
# Extract log probs and ground truth at digit positions
digit_positions_log_probs = digit_log_probs[digit_mask]  # [num_digits, 10]
digit_positions_ground_truth = digit_ground_truth_tensor[digit_mask]  # [num_digits, 10]

# Calculate NTL loss: -log(P(true_digit))
digit_losses = -(digit_positions_log_probs * digit_positions_ground_truth).sum(dim=-1)
ntl_loss = digit_losses.mean().item()
```

**Characteristics:**
- **Direct probability-based**: Uses raw log probabilities from the model
- **Classification-focused**: Treats each digit as an independent classification task
- **Gradient-friendly**: Provides direct gradients to improve digit token probabilities
- **Mathematically principled**: Standard cross-entropy loss for digit positions

### **Method 2: Confidence-Based Loss (`custom_ntl_final.py`)**

This method converts probabilities to confidence scores and uses negative log confidence as the loss metric.

**Mathematical Formulation:**

```python
# For each digit position i with ground truth digit d_i:
# 1. Convert log probabilities to probabilities
P_i = exp(logits_i) / Σ(exp(logits_i[j]) for j in 0..9)  # Softmax

# 2. Calculate confidence as probability of true digit
confidence_i = P_i[d_i]

# 3. Average confidence across all digit positions
avg_confidence = (1/N) × Σ(confidence_i for i in digit_positions)

# 4. Convert confidence to loss
ntl_loss = -log(max(avg_confidence, ε))  # ε = 1e-8 for numerical stability
```

**Implementation:**
```python
# Convert log probs to probabilities
digit_probs = torch.exp(digit_positions_log_probs)  # [num_digits, 10]

# Calculate confidence as probability of true digit
confidences = (digit_probs * digit_positions_ground_truth).sum(dim=-1)  # [num_digits]

# Average confidence across all positions
avg_confidence = confidences.mean().item()

# Convert to loss
ntl_loss = -math.log(max(avg_confidence, 1e-8))
```

**Characteristics:**
- **Confidence-focused**: Measures model's confidence in correct digit predictions
- **Averaged approach**: Uses mean confidence rather than sum of losses
- **Intuitive interpretation**: Higher confidence → lower loss
- **Robust to outliers**: Averaging smooths extreme predictions

---

## **Detailed Mathematical Comparison**

### **Loss Behavior Analysis**

Let's compare both methods with concrete examples:

**Example 1: Perfect Prediction**
```python
# Scenario: Model perfectly predicts digits "123"
# P(1) = 1.0, P(2) = 1.0, P(3) = 1.0

# Method 1 (Negative Log-Likelihood):
loss_1 = -log(1.0) = 0.0
loss_2 = -log(1.0) = 0.0  
loss_3 = -log(1.0) = 0.0
ntl_loss = (0.0 + 0.0 + 0.0) / 3 = 0.0

# Method 2 (Confidence-Based):
confidence = (1.0 + 1.0 + 1.0) / 3 = 1.0
ntl_loss = -log(1.0) = 0.0

# Result: Both methods give identical loss = 0.0
```

**Example 2: Moderate Confidence**
```python
# Scenario: Model moderately confident on digits "186"
# P(1) = 0.7, P(8) = 0.6, P(6) = 0.8

# Method 1 (Negative Log-Likelihood):
loss_1 = -log(0.7) = 0.357
loss_8 = -log(0.6) = 0.511
loss_6 = -log(0.8) = 0.223
ntl_loss = (0.357 + 0.511 + 0.223) / 3 = 0.364

# Method 2 (Confidence-Based):
confidence = (0.7 + 0.6 + 0.8) / 3 = 0.7
ntl_loss = -log(0.7) = 0.357

# Result: Method 1 = 0.364, Method 2 = 0.357 (similar but different)
```

**Example 3: Mixed Performance**
```python
# Scenario: One very confident, one very uncertain
# P(1) = 0.95, P(8) = 0.1, P(6) = 0.8

# Method 1 (Negative Log-Likelihood):
loss_1 = -log(0.95) = 0.051
loss_8 = -log(0.1) = 2.303   # High penalty for low confidence
loss_6 = -log(0.8) = 0.223
ntl_loss = (0.051 + 2.303 + 0.223) / 3 = 0.859

# Method 2 (Confidence-Based):
confidence = (0.95 + 0.1 + 0.8) / 3 = 0.617
ntl_loss = -log(0.617) = 0.483

# Result: Method 1 = 0.859, Method 2 = 0.483 (Method 1 more sensitive to outliers)
```

### **Key Mathematical Differences**

| Aspect | Method 1 (NLL) | Method 2 (Confidence) |
|--------|----------------|----------------------|
| **Sensitivity** | High - each bad prediction heavily penalized | Moderate - averaging smooths outliers |
| **Gradient Signal** | Direct per-position gradients | Averaged gradients across positions |
| **Outlier Handling** | Sensitive to individual poor predictions | Robust due to averaging |
| **Mathematical Form** | `Σ(-log(p_i))/N` | `-log(Σ(p_i)/N)` |
| **Training Focus** | Individual digit accuracy | Overall prediction confidence |

---

## **Reward Function Integration**

Both methods integrate with the reward system using the same exponential transformation:

```python
# Common reward formula for both methods:
reward = exact_match × exp(-ntl_loss / τ)

# Where:
# - exact_match ∈ {0, 1} for custom_ntl_all.py
# - exact_match = 1 (always) for custom_ntl_final.py  
# - τ (tau) is the temperature parameter
```

### **Temperature Parameter Effects**

The temperature τ controls the sensitivity of the exponential reward function:

**Mathematical Analysis:**
```python
# For a fixed ntl_loss = 1.0:

τ = 0.5:  reward = exp(-1.0/0.5) = exp(-2.0) = 0.135  # Harsh penalty
τ = 1.0:  reward = exp(-1.0/1.0) = exp(-1.0) = 0.368  # Standard
τ = 2.0:  reward = exp(-1.0/2.0) = exp(-0.5) = 0.607  # Gentle penalty
τ = 5.0:  reward = exp(-1.0/5.0) = exp(-0.2) = 0.819  # Very forgiving
```

**Gradient Impact:**
```python
# Derivative of reward w.r.t. ntl_loss:
∂reward/∂ntl_loss = -(1/τ) × exp(-ntl_loss/τ)

# Lower τ → Higher gradient magnitude → More aggressive learning
# Higher τ → Lower gradient magnitude → More stable learning
```

---

## **Training Examples and Expected Behavior**

### **Example Training Scenarios**

**Scenario 1: Correct Answer with Good NTL**
```python
# Input: "#### 186", Ground Truth: "186"
# NTL: P(1)=0.9, P(8)=0.8, P(6)=0.9

# Method 1 (NLL):
ntl_loss = -(log(0.9) + log(0.8) + log(0.9))/3 = 0.117
reward = 1.0 × exp(-0.117/1.0) = 0.890

# Method 2 (Confidence):  
confidence = (0.9 + 0.8 + 0.9)/3 = 0.867
ntl_loss = -log(0.867) = 0.143
reward = 1.0 × exp(-0.143/1.0) = 0.867

# Both methods give high rewards for good performance
```

**Scenario 2: Correct Answer with Poor NTL**
```python
# Input: "#### 186", Ground Truth: "186"  
# NTL: P(1)=0.4, P(8)=0.3, P(6)=0.4

# Method 1 (NLL):
ntl_loss = -(log(0.4) + log(0.3) + log(0.4))/3 = 1.084
reward = 1.0 × exp(-1.084/1.0) = 0.338

# Method 2 (Confidence):
confidence = (0.4 + 0.3 + 0.4)/3 = 0.367
ntl_loss = -log(0.367) = 1.002  
reward = 1.0 × exp(-1.002/1.0) = 0.367

# Both methods penalize poor digit confidence
```

**Scenario 3: Wrong Answer with Good NTL**
```python
# Input: "#### 187", Ground Truth: "186" (wrong answer)
# NTL: P(1)=0.9, P(8)=0.8, P(7)=0.9 (confident but wrong final answer)

# Method 1 (custom_ntl_all.py):
reward = 0.0 × exp(-ntl_loss) = 0.0  # No reward for wrong answer

# Method 2 (custom_ntl_final.py):  
# Uses final answer extraction, but still wrong
reward = 0.0 × exp(-ntl_loss) = 0.0  # No reward for wrong answer

# Key insight: Both methods require correctness for full reward
```

---

## **Implementation Recommendations**

### **When to Use Method 1 (Negative Log-Likelihood)**

✅ **Recommended for:**
- Models that need strong per-digit feedback
- Training scenarios where individual digit accuracy is crucial
- When you want maximum gradient signal for digit token improvement
- Mathematical reasoning tasks requiring precise numerical computation

✅ **Advantages:**
- Direct gradient flow to each digit position
- Mathematically principled (standard cross-entropy)
- High sensitivity to digit prediction quality
- Well-established theoretical foundation

⚠️ **Considerations:**
- More sensitive to outlier predictions
- Can be harsh on mixed-quality predictions
- May require careful temperature tuning

### **When to Use Method 2 (Confidence-Based)**

✅ **Recommended for:**
- Models that need robust, averaged feedback
- Training scenarios prioritizing overall confidence
- When dealing with noisy or inconsistent digit predictions
- Applications requiring stable gradient signals

✅ **Advantages:**
- Robust to outlier predictions due to averaging
- Intuitive confidence-based interpretation
- Smoother training dynamics
- Less sensitive to individual prediction errors

⚠️ **Considerations:**
- May dilute strong gradient signals
- Less direct connection to individual digit improvement
- Averaging may mask specific weaknesses

### **Hybrid Approach Recommendation**

For optimal results, consider curriculum learning:

```python
# Stage 1: Use Method 2 for stable initial learning
if training_epoch < 5:
    use_confidence_based_loss()  # Method 2
    
# Stage 2: Switch to Method 1 for fine-grained improvement  
else:
    use_negative_log_likelihood_loss()  # Method 1
```

---

This comprehensive mathematical analysis demonstrates that both methods provide valid approaches to NTL-based training, with Method 1 offering more direct gradient signals and Method 2 providing more robust, averaged feedback. The choice depends on your specific training requirements and model behavior characteristics.

---

## Testing and Validation

### **Run Comprehensive Tests**

```bash
# Test both NTL modes
python test_both_ntl_modes.py

# Test individual components  
python test_custom_ntl.py
```

### **Expected Test Results**

| Scenario | Final Mode | All Mode | Interpretation |
|----------|------------|----------|----------------|
| Correct + Good NTL | 0.607 | 0.607 | Same reward for both modes |
| Correct + Poor NTL | 0.135 | 0.135 | Same reward for both modes |
| **Wrong + Good NTL** | **0.607** | **0.000** | Key difference: Final rewards NTL quality |
| Wrong + Poor NTL | 0.135 | 0.000 | Final still rewards some NTL quality |
| Correct + No NTL | 0.000 | 1.000 | Final needs NTL, All falls back to exact match |

---

## Performance Expectations

### **Final Answer Mode Benefits**
- ✅ **Continuous Learning**: Model receives feedback even for wrong answers
- ✅ **Numerical Focus**: Specifically improves digit-level prediction quality
- ✅ **Robust Training**: Less susceptible to reward sparsity issues
- ✅ **Quality Improvement**: Encourages better probability distributions for digits

### **All Tokens Mode Benefits**  
- ✅ **Accuracy Preservation**: Maintains exact match requirement
- ✅ **Comprehensive Coverage**: Improves reasoning throughout entire solution
- ✅ **Traditional + NTL**: Combines benefits of both approaches
- ✅ **Quality Assurance**: Only rewards high-quality numerical reasoning

### **Training Recommendations**

| Use Final Mode When: | Use All Mode When: |
|----------------------|-------------------|
| Model struggles with digit prediction | Model has reasonable accuracy but poor reasoning quality |
| Want to improve numerical understanding | Want to maintain accuracy while improving quality |
| Dealing with reward sparsity | Have sufficient exact match performance |
| Focus on digit-level capabilities | Need comprehensive solution quality |

---

## Tokenization Compatibility

### **Qwen2.5 Tokenization Analysis**

**✅ Verified Compatible**: All numbers 0-100 are either single digit tokens or composed of individual digit tokens.

| Number Type | Tokenization | NTL Compatibility |
|-------------|--------------|-------------------|
| Single digits (0-9) | Single tokens [15-24] | ✅ Perfect |
| Two digits (10-99) | Two digit tokens | ✅ Perfect |  
| Three digits (100) | Three digit tokens | ✅ Perfect |

**Examples:**
- "7" → [22] (single token for digit 7)
- "18" → [16, 23] (tokens for digits 1 and 8)
- "186" → [16, 23, 21] (tokens for digits 1, 8, and 6)
- "100" → [16, 15, 15] (tokens for digits 1, 0, and 0)

---

## Implementation Details

### **Integration Points**

1. **NTL Core** (`ntl_local/ntl_core.py`): Calculates digit-level NTL loss
2. **Actor Extension** (`dp_actor_ntl.py`): Extracts digit information during forward pass  
3. **Reward Functions** (`custom_ntl_*.py`): Converts NTL loss to reward signals
4. **Training Scripts**: Orchestrate the entire NTL-enabled training process

### **Data Flow**

```
Model Forward Pass
    ↓
Extract Digit Log Probabilities (NTLDigitExtractor)
    ↓  
Calculate NTL Loss (ntl_core.py)
    ↓
Pass to Reward Function (custom_ntl_*.py)
    ↓
Apply Temperature: exp(-loss/Tau)
    ↓
Return Final Reward Score
```

---

## Advanced Configuration

### **Custom Temperature Schedules**

You can implement temperature scheduling in the training scripts:

```bash
# Example: Start with high temperature, gradually decrease
if [ $EPOCH -lt 5 ]; then
    TAU=2.0  # More forgiving early on
elif [ $EPOCH -lt 10 ]; then  
    TAU=1.0  # Standard sensitivity
else
    TAU=0.5  # More aggressive later
fi

custom_reward_function.tau=$TAU
```

### **Debugging and Monitoring**

Monitor these metrics during training:

- **Average NTL Loss**: Should decrease over time
- **Digit Accuracy**: Percentage of correctly predicted digits
- **Reward Distribution**: Should shift toward higher values
- **Temperature Effect**: Compare results with different Tau values

---

## Troubleshooting

### **Common Issues**

1. **No NTL Info Available**: Check that NTL-enabled actor is being used
2. **NTL Loss is inf**: Verify digit positions are correctly identified
3. **Poor Reward Signal**: Try adjusting temperature parameter
4. **Training Instability**: Consider higher Tau for more stable gradients

### **Validation Checks**

```bash
# Verify NTL integration
python -c "
from custom_ntl_final import compute_score
result = compute_score('test', 'Step: #### 18', '18', ntl_info={'ntl_loss': 0.5}, tau=1.0)
print(f'Test reward: {result}')
"
```

---

## Future Enhancements

### **Potential Improvements**

1. **Adaptive Temperature**: Automatically adjust Tau based on training progress
2. **Position-Weighted Loss**: Higher importance for more significant digits
3. **Curriculum Learning**: Progressive difficulty in numerical reasoning
4. **Multi-Dataset Training**: Extend beyond GSM8K to other math datasets

### **Research Directions**

1. **Comparative Analysis**: Systematic comparison of Final vs All modes
2. **Temperature Optimization**: Find optimal Tau values for different scenarios  
3. **Intermediate Step Evaluation**: Extend NTL to reasoning step evaluation
4. **Cross-Domain Transfer**: Apply NTL insights to other numerical tasks

---

## References

- **Original NTL Paper**: "Regress, Don't Guess – A Regression-like Loss on Number Tokens for Language Models"
- **VERL Framework**: https://github.com/volcengine/verl
- **Implementation Repository**: https://github.com/tum-ai/number-token-loss

---

## Conclusion

The NTL integration provides two complementary approaches to improving numerical reasoning in language models. The Final Answer mode offers continuous learning signals for digit prediction quality, while the All Tokens mode maintains accuracy requirements while enhancing overall reasoning quality. The configurable temperature parameter allows fine-tuning the reward sensitivity for optimal training results.

**Quick Start:**
1. Choose your mode: Final Answer (continuous) or All Tokens (accuracy-focused)
2. Set appropriate temperature: Tau=1.0 (standard), Tau=0.5 (aggressive), Tau=2.0 (forgiving)
3. Run training script: `./train_gsm8k_qwen_ppo_ntl_final_answer.sh` or `./train_gsm8k_qwen_ppo_ntl.sh`
4. Monitor NTL loss and reward metrics for training progress

---

*Implementation ready for production training with comprehensive testing validation.*