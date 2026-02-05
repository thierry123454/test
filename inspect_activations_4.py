import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set model to evaluation mode
model.eval()

# Define original and modified prompts
original_prompt = "The capital of France is Paris, and the capital of Germany is"
modified_prompt = "The capital of France is Paris, and the capital of Spain is"  # Replace "Germany" with "Spain"

print("=" * 80)
print("ABLATION STUDY: IMPACT OF TOKEN CHANGES ON PREDICTIONS")
print("=" * 80)
print()

print("Original prompt:")
print(f"  '{original_prompt}'")
print()

print("Modified prompt:")
print(f"  '{modified_prompt}'")
print()

print("Change: 'Germany' → 'Spain'")
print()

# ===== ANALYZE ORIGINAL PROMPT =====
print("=" * 80)
print("ORIGINAL PROMPT ANALYSIS")
print("=" * 80)
print()

# Tokenize original prompt
original_input_ids = tokenizer.encode(original_prompt, return_tensors='pt')
original_tokens = tokenizer.convert_ids_to_tokens(original_input_ids[0])

print(f"Tokens: {original_tokens}")
print(f"Number of tokens: {len(original_tokens)}")
print()

# Forward pass
with torch.no_grad():
    original_outputs = model(original_input_ids)
    original_logits = original_outputs.logits

# Get next-token logits (for the last position)
original_next_token_logits = original_logits[:, -1, :]

# Compute probabilities
original_probs = F.softmax(original_next_token_logits, dim=-1)[0]
original_log_probs = F.log_softmax(original_next_token_logits, dim=-1)[0]

# Get top-5 predictions
top_5_probs, top_5_indices = torch.topk(original_probs, 5)

print("Top-5 next-token predictions:")
print("-" * 80)
print(f"{'Rank':<6} {'Token':<20} {'Probability':<15} {'Log Prob':<12}")
print("-" * 80)

for rank, (prob, idx) in enumerate(zip(top_5_probs, top_5_indices), 1):
    token = tokenizer.decode([idx.item()])
    print(f"{rank:<6} {repr(token):<20} {prob.item():<15.6f} {original_log_probs[idx].item():<12.4f}")

print()

# ===== ANALYZE MODIFIED PROMPT =====
print("=" * 80)
print("MODIFIED PROMPT ANALYSIS")
print("=" * 80)
print()

# Tokenize modified prompt
modified_input_ids = tokenizer.encode(modified_prompt, return_tensors='pt')
modified_tokens = tokenizer.convert_ids_to_tokens(modified_input_ids[0])

print(f"Tokens: {modified_tokens}")
print(f"Number of tokens: {len(modified_tokens)}")
print()

# Forward pass
with torch.no_grad():
    modified_outputs = model(modified_input_ids)
    modified_logits = modified_outputs.logits

# Get next-token logits (for the last position)
modified_next_token_logits = modified_logits[:, -1, :]

# Compute probabilities
modified_probs = F.softmax(modified_next_token_logits, dim=-1)[0]
modified_log_probs = F.log_softmax(modified_next_token_logits, dim=-1)[0]

# Get top-5 predictions
top_5_probs_mod, top_5_indices_mod = torch.topk(modified_probs, 5)

print("Top-5 next-token predictions:")
print("-" * 80)
print(f"{'Rank':<6} {'Token':<20} {'Probability':<15} {'Log Prob':<12}")
print("-" * 80)

for rank, (prob, idx) in enumerate(zip(top_5_probs_mod, top_5_indices_mod), 1):
    token = tokenizer.decode([idx.item()])
    print(f"{rank:<6} {repr(token):<20} {prob.item():<15.6f} {modified_log_probs[idx].item():<12.4f}")

print()

# ===== SIDE-BY-SIDE COMPARISON =====
print("=" * 80)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 80)
print()

print(f"{'Rank':<6} {'Original Token':<20} {'Prob':<12} {'Modified Token':<20} {'Prob':<12} {'Change':<12}")
print("-" * 80)

for rank in range(5):
    orig_token = tokenizer.decode([top_5_indices[rank].item()])
    orig_prob = top_5_probs[rank].item()
    
    mod_token = tokenizer.decode([top_5_indices_mod[rank].item()])
    mod_prob = top_5_probs_mod[rank].item()
    
    change = mod_prob - orig_prob
    change_str = f"{change:+.6f}"
    
    print(f"{rank+1:<6} {repr(orig_token):<20} {orig_prob:<12.6f} {repr(mod_token):<20} {mod_prob:<12.6f} {change_str:<12}")

print()

# ===== SPECIFIC TOKEN PROBABILITY CHANGES =====
print("=" * 80)
print("PROBABILITY CHANGES FOR SPECIFIC TOKENS")
print("=" * 80)
print()

# Check probabilities for specific expected answers
expected_tokens = ["Berlin", " Berlin", "Madrid", " Madrid", "Barcelona", " Barcelona"]

print("How did probabilities change for expected capital cities?")
print("-" * 80)
print(f"{'Token':<20} {'Original Prob':<15} {'Modified Prob':<15} {'Change':<15}")
print("-" * 80)

for token_str in expected_tokens:
    token_id = tokenizer.encode(token_str, add_special_tokens=False)
    if len(token_id) > 0:
        token_id = token_id[0]
        orig_prob = original_probs[token_id].item()
        mod_prob = modified_probs[token_id].item()
        change = mod_prob - orig_prob
        print(f"{repr(token_str):<20} {orig_prob:<15.6f} {mod_prob:<15.6f} {change:+15.6f}")

print()

# ===== ENTROPY COMPARISON =====
print("=" * 80)
print("PREDICTION UNCERTAINTY (ENTROPY)")
print("=" * 80)
print()

# Calculate entropy: H = -sum(p * log(p))
original_entropy = -(original_probs * original_log_probs).sum().item()
modified_entropy = -(modified_probs * modified_log_probs).sum().item()

print(f"Original prompt entropy: {original_entropy:.4f}")
print(f"Modified prompt entropy: {modified_entropy:.4f}")
print(f"Change in entropy:       {modified_entropy - original_entropy:+.4f}")
print()

if modified_entropy > original_entropy:
    print("→ Model is MORE uncertain with the modified prompt")
elif modified_entropy < original_entropy:
    print("→ Model is LESS uncertain with the modified prompt")
else:
    print("→ Model has SAME uncertainty")

print()

# ===== SUMMARY =====
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print("Key Findings:")
print()

# Check if top prediction changed
orig_top_token = tokenizer.decode([top_5_indices[0].item()])
mod_top_token = tokenizer.decode([top_5_indices_mod[0].item()])

if orig_top_token == mod_top_token:
    print(f"✓ Top prediction UNCHANGED: '{orig_top_token}'")
    print(f"  Probability changed from {top_5_probs[0].item():.6f} to {top_5_probs_mod[0].item():.6f}")
else:
    print(f"✗ Top prediction CHANGED:")
    print(f"  Original: '{orig_top_token}' ({top_5_probs[0].item():.6f})")
    print(f"  Modified: '{mod_top_token}' ({top_5_probs_mod[0].item():.6f})")

print()

# Check if any of the top-5 changed
orig_set = set([tokenizer.decode([idx.item()]) for idx in top_5_indices])
mod_set = set([tokenizer.decode([idx.item()]) for idx in top_5_indices_mod])

new_tokens = mod_set - orig_set
removed_tokens = orig_set - mod_set

if new_tokens:
    print(f"New tokens in top-5: {new_tokens}")
if removed_tokens:
    print(f"Removed from top-5: {removed_tokens}")

print()
print("Interpretation:")
print("  Changing 'Germany' to 'Spain' affects the model's prediction because")
print("  the expected answer changes from 'Berlin' to 'Madrid' or 'Barcelona'.")
print("  This demonstrates how context influences next-token predictions.")

