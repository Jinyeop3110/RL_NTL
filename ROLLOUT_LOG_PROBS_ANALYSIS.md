# rollout_log_probs Structure and Usage Analysis

**Date**: July 30, 2025  
**Purpose**: Understanding the structure and potential of rollout_log_probs for NTL computation

---

## What is rollout_log_probs?

### **Definition**
`rollout_log_probs` is a tensor containing the **log probabilities of the tokens that were actually selected** during sequence generation (rollout phase) in PPO training.

### **Structure**
```python
rollout_log_probs: torch.Tensor  # Shape: [batch_size, response_length]
```

- **Dimensions**: `[batch_size, response_length]`
- **Data Type**: `torch.float32`
- **Values**: Log probabilities (negative values, closer to 0 = higher probability)
- **Correspondence**: `rollout_log_probs[i, j]` = log P(token selected at position j in sequence i)

---

## How rollout_log_probs is Created

### **In VLLM Rollout** (`vllm_rollout_spmd.py:337-338`)

```python
for i, logprob in enumerate(output.outputs[sample_id].logprobs):
    curr_log_prob.append(logprob[response_ids[i]].logprob)  # ← Key line
```

**What this means:**
1. `output.outputs[sample_id].logprobs` contains the **full probability distribution** for each generated position
2. `response_ids[i]` is the **token ID that was actually selected**
3. `logprob[response_ids[i]].logprob` extracts the **log probability of the selected token**

### **Key Insight**: VLLM Actually Has Full Distributions!

The crucial discovery is that **VLLM's `logprobs` object contains the complete probability distribution**, not just the selected token's probability. From line 338:

- `logprob` is a dict-like object with entries for **all tokens in vocabulary**
- `logprob[token_id]` gives the probability info for any token_id
- `logprob[response_ids[i]]` is just extracting the probability of the **selected** token

---

## What We Currently Extract vs. What We Could Extract

### **Current Implementation (Limited)**
```python
# What we currently do
rollout_log_probs[batch_idx, pos].item()  # Log prob of selected token only
```

**Information Available:**
- ✅ Log probability of selected digit tokens
- ✅ Confidence measure for generated digits
- ❌ Cannot compute regression-based NTL loss
- ❌ No comparison with alternative digit probabilities

### **Potential Full Implementation (Complete NTL)**

If we modify the rollout to capture **digit probability distributions**:

```python
# What we could do (requires rollout modification)
for pos, logprob_dict in enumerate(output.outputs[sample_id].logprobs):
    digit_log_probs = []
    for digit in range(10):
        digit_token_id = digit_to_token_map[digit]
        if digit_token_id in logprob_dict:
            digit_log_probs.append(logprob_dict[digit_token_id].logprob)
        else:
            digit_log_probs.append(-float('inf'))  # Not a valid token
    
    # Now we have P(digit=0), P(digit=1), ..., P(digit=9) for position pos
    digit_probs = torch.softmax(torch.tensor(digit_log_probs), dim=0)
    expected_digit = torch.sum(torch.arange(10) * digit_probs)  # E[digit]
    ntl_loss = (true_digit - expected_digit) ** 2  # True NTL loss!
```

---

## Current NTL Information Structure (Rollout-Based)

### **What We Extract Now**
```python
ntl_info = {
    # Basic identification
    'digit_positions': List[List[int]],           # Where digits occur in responses
    'digit_ground_truth': List[List[int]],        # Selected digit values (0-9)
    'digit_log_probs_selected': List[List[float]], # Log P(selected digit)
    
    # Confidence metrics (not true NTL)
    'ntl_loss': float,                           # -avg_digit_log_prob (proxy)
    'avg_digit_log_prob': float,                 # Average confidence
    'total_digits': int,                         # Count of digit tokens
    
    # Metadata
    'method': 'simplified_from_rollout_log_probs'
}
```

### **Example Data**
```python
# Example for response: "The answer is 186"
# Tokens: ["The", "answer", "is", "1", "8", "6"]
# Positions: [3, 4, 5] contain digits [1, 8, 6]

ntl_info = {
    'digit_positions': [[3, 4, 5]],              # Positions of "1", "8", "6"
    'digit_ground_truth': [[1, 8, 6]],           # The actual digit values
    'digit_log_probs_selected': [[-0.05, -0.1, -0.2]], # Log P("1"), Log P("8"), Log P("6")
    'ntl_loss': 0.117,                          # -(-0.05 + -0.1 + -0.2)/3 = 0.117
    'avg_digit_log_prob': -0.117,               # Average confidence
    'total_digits': 3
}
```

---

## Advantages of Current Approach

### **✅ Immediate Benefits**
1. **No Code Changes**: Uses existing `rollout_log_probs` 
2. **Digit Confidence**: Measures how confident model was about digit choices
3. **Position Tracking**: Knows where digits occur in sequences
4. **Reward Integration**: Can pass confidence info to reward functions

### **✅ Meaningful Signal**
- High confidence digits: `log_prob ≈ 0` (e.g., -0.01) → Model very sure
- Low confidence digits: `log_prob << 0` (e.g., -2.3) → Model uncertain
- Reward functions can use this to prefer confident digit predictions

---

## Limitations of Current Approach

### **❌ Not True NTL Loss**
- Cannot compute `E[digit] = Σ(digit × P(digit))` (missing P(digit) for digit≠selected)
- Cannot do regression-based comparison of expected vs. true digits
- Missing the core NTL insight about treating digits as continuous values

### **❌ No Ground Truth Comparison**  
- Treats all generated digits as "correct" 
- Cannot measure actual accuracy against dataset answers
- Cannot distinguish between right/wrong confident predictions

### **❌ Limited Learning Signal**
- Only provides confidence feedback, not correctness feedback
- Model doesn't learn "digit 7 was wrong, should have been 6"
- Missing the regression-based learning that NTL provides

---

## Path to Full NTL Implementation

### **Option 1: Modify Rollout to Capture Digit Distributions**

Modify VLLM rollout to extract **full digit probability distributions**:

```python
# In vllm_rollout_spmd.py
if position_has_digit_token(pos):
    digit_log_probs = extract_digit_distribution(logprob_dict, tokenizer)
    # Store digit_log_probs[batch][pos] = [P(0), P(1), ..., P(9)]
```

**Pros:** Full NTL computation capability
**Cons:** Requires modifying rollout code, increased memory usage

### **Option 2: Use Existing NTL Actors During compute_log_prob**

Use the existing `dp_actor_ntl.py` / `fsdp_actor_ntl.py` during the `compute_log_prob` phase:

```python
# These actors can extract full logits during forward pass
# Then compute complete NTL loss with ground truth comparison
```

**Pros:** Leverages existing NTL implementation
**Cons:** Requires coordinating with actor modifications

### **Option 3: Hybrid Approach**

1. Use current rollout-based confidence for **generation feedback**
2. Use actor-based full NTL during **policy updates**

**Pros:** Best of both worlds
**Cons:** More complex implementation

---

## Recommendations

### **For Immediate Use**
The current rollout-based approach provides **useful digit confidence signals** that can improve training:

```python
# Reward function can use confidence
if ntl_info and ntl_info['avg_digit_log_prob'] > -0.5:  # High confidence
    reward *= 1.1  # Bonus for confident digit predictions
```

### **For Full NTL Benefits**
To get complete NTL regression-based learning:

1. **Modify VLLM rollout** to capture digit probability distributions at generation time
2. **Extract full NTL loss** using the complete P(digit) distributions  
3. **Compare against ground truth** from dataset for true accuracy

### **Incremental Path**
1. ✅ **Phase 1**: Use current confidence-based approach (working now)
2. 🔄 **Phase 2**: Add ground truth comparison for accuracy measurement
3. 🚀 **Phase 3**: Implement full probability distribution capture for true NTL

---

## Code Examples

### **Current Usage (Confidence-Based)**
```python
def compute_confidence_reward(ntl_info, base_reward):
    if ntl_info and ntl_info['total_digits'] > 0:
        confidence = math.exp(ntl_info['avg_digit_log_prob'])  # Convert to [0,1]
        return base_reward * (0.5 + 0.5 * confidence)  # Scale by confidence
    return base_reward
```

### **Potential Full NTL Usage**
```python
def compute_full_ntl_reward(digit_distributions, ground_truth_digits):
    ntl_losses = []
    for pos, (dist, true_digit) in enumerate(zip(digit_distributions, ground_truth_digits)):
        expected_digit = sum(digit * prob for digit, prob in enumerate(dist))
        loss = (true_digit - expected_digit) ** 2
        ntl_losses.append(loss)
    
    avg_ntl_loss = sum(ntl_losses) / len(ntl_losses)
    return math.exp(-avg_ntl_loss)  # Convert to reward
```

---

## Conclusion

The `rollout_log_probs` provides a **valuable foundation** for NTL-like improvements, offering digit confidence signals that can enhance training. While it doesn't provide full NTL regression capability, it's an excellent **starting point** that requires no actor modifications and leverages existing infrastructure.

The **full NTL implementation** would require capturing complete digit probability distributions during rollout, but the current approach already provides meaningful feedback for numerical reasoning improvements.