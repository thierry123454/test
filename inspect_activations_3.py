import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set model to evaluation mode
model.eval()

# Define a prompt
prompt = "The capital of France is Paris, and the capital of Germany is"

print("=" * 80)
print("ATTENTION PATTERN ANALYSIS")
print("=" * 80)
print()

print(f"Prompt: '{prompt}'")
print()

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

print(f"Number of tokens: {len(tokens)}")
print(f"Tokens: {tokens}")
print()

# Forward pass with output_attentions=True
with torch.no_grad():
    outputs = model(input_ids, output_attentions=True)
    attentions = outputs.attentions

print("=" * 80)
print("ATTENTION STRUCTURE")
print("=" * 80)
print()

num_layers = len(attentions)
print(f"Number of layers with attention: {num_layers}")
print()

# Inspect shape of attention tensors
print("Shape of attention tensors:")
print("-" * 80)
print(f"{'Layer':<10} {'Shape':<30} {'Description'}")
print("-" * 80)

for l in range(num_layers):
    shape = attentions[l].shape
    print(f"Layer {l+1:<4} {str(tuple(shape)):<30} (batch, heads, seq, seq)")

print()
print(f"Format: (batch_size, num_heads, sequence_length, sequence_length)")
print(f"Number of attention heads: {attentions[0].shape[1]}")
print()

# ===== ANALYZE LAST TOKEN ATTENTION =====
print("=" * 80)
print("ATTENTION WEIGHTS FOR LAST TOKEN")
print("=" * 80)
print()

last_token_pos = len(tokens) - 1
last_token = tokens[last_token_pos]

print(f"Analyzing attention for last token (position {last_token_pos}): '{last_token}'")
print()

# Pick a specific layer and head to analyze
analyze_layer = 11  # Last layer (0-indexed: layer 12)
analyze_head = 0    # First head

print(f"Analyzing Layer {analyze_layer + 1}, Head {analyze_head + 1}")
print()

# Get attention weights for the last token
# Shape: (batch, heads, seq, seq)
# We want: attentions[layer][batch, head, last_token_pos, :]
attention_weights = attentions[analyze_layer][0, analyze_head, last_token_pos, :]

print(f"Attention weights shape: {attention_weights.shape}")
print(f"Sum of attention weights: {attention_weights.sum().item():.6f} (should be ~1.0)")
print()

# Get top-k tokens that receive highest attention
top_k = 10
top_weights, top_indices = torch.topk(attention_weights, min(top_k, len(tokens)))

print(f"Top {min(top_k, len(tokens))} tokens receiving highest attention:")
print("-" * 80)
print(f"{'Rank':<6} {'Position':<10} {'Token':<20} {'Attention Weight':<20}")
print("-" * 80)

for rank, (weight, idx) in enumerate(zip(top_weights, top_indices), 1):
    token = tokens[idx.item()]
    pos = idx.item()
    attn_weight = weight.item()
    marker = "← LAST TOKEN" if pos == last_token_pos else ""
    print(f"{rank:<6} {pos:<10} {repr(token):<20} {attn_weight:<20.6f} {marker}")

print()

# ===== ANALYZE MULTIPLE HEADS =====
print("=" * 80)
print("ATTENTION PATTERNS ACROSS DIFFERENT HEADS (Layer 12)")
print("=" * 80)
print()

num_heads = attentions[analyze_layer].shape[1]
print(f"Analyzing all {num_heads} heads for the last token")
print()

for head in range(num_heads):
    attention_weights = attentions[analyze_layer][0, head, last_token_pos, :]
    
    # Get top-3 tokens for this head
    top_3_weights, top_3_indices = torch.topk(attention_weights, 3)
    
    top_tokens = [tokens[idx.item()] for idx in top_3_indices]
    top_weights_list = [w.item() for w in top_3_weights]
    
    print(f"Head {head + 1:2d}: Top-3 = {top_tokens[0]:>15} ({top_weights_list[0]:.4f}), "
          f"{top_tokens[1]:>15} ({top_weights_list[1]:.4f}), "
          f"{top_tokens[2]:>15} ({top_weights_list[2]:.4f})")

print()

# ===== ANALYZE ATTENTION TO SPECIFIC TOKENS =====
print("=" * 80)
print("ATTENTION TO SEMANTICALLY IMPORTANT TOKENS")
print("=" * 80)
print()

# Find positions of important tokens
important_tokens = {}
for i, token in enumerate(tokens):
    if 'France' in token:
        important_tokens['France'] = i
    elif 'Paris' in token:
        important_tokens['Paris'] = i
    elif 'Germany' in token:
        important_tokens['Germany'] = i
    elif 'capital' in token:
        if 'capital_1' not in important_tokens:
            important_tokens['capital_1'] = i
        else:
            important_tokens['capital_2'] = i

print(f"Last token '{last_token}' attention to important tokens (Layer {analyze_layer + 1}, Head {analyze_head + 1}):")
print("-" * 80)

for token_name, pos in important_tokens.items():
    attn_weight = attentions[analyze_layer][0, analyze_head, last_token_pos, pos].item()
    print(f"  {token_name:<15} (pos {pos}): {attn_weight:.6f}")

print()

# ===== LAYER-WISE ATTENTION ANALYSIS =====
print("=" * 80)
print("ATTENTION TO 'Germany' ACROSS LAYERS (Head 1)")
print("=" * 80)
print()

if 'Germany' in important_tokens:
    germany_pos = important_tokens['Germany']
    print(f"Tracking how much the last token attends to 'Germany' (pos {germany_pos}) across layers")
    print()
    
    print(f"{'Layer':<10} {'Attention Weight':<20}")
    print("-" * 80)
    
    for l in range(num_layers):
        attn_weight = attentions[l][0, 0, last_token_pos, germany_pos].item()
        layer_name = f"Layer {l + 1}"
        print(f"{layer_name:<10} {attn_weight:<20.6f}")
    
    print()

# ===== FULL ATTENTION DISTRIBUTION =====
print("=" * 80)
print("FULL ATTENTION DISTRIBUTION (Layer 12, Head 1)")
print("=" * 80)
print()

print(f"All attention weights for last token '{last_token}':")
print("-" * 80)
print(f"{'Position':<10} {'Token':<20} {'Attention Weight':<20} {'Bar Chart'}")
print("-" * 80)

attention_weights = attentions[analyze_layer][0, analyze_head, last_token_pos, :]

for pos in range(len(tokens)):
    token = tokens[pos]
    weight = attention_weights[pos].item()
    bar_length = int(weight * 50)  # Scale to 50 chars max
    bar = '█' * bar_length
    print(f"{pos:<10} {repr(token):<20} {weight:<20.6f} {bar}")

print()

# ===== SUMMARY =====
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print(f"Model: GPT-2")
print(f"Number of layers: {num_layers}")
print(f"Number of attention heads per layer: {num_heads}")
print(f"Attention tensor shape: (batch, heads, seq_len, seq_len)")
print()

print("Key Observations:")
print(f"  - Each attention head can focus on different aspects of the input")
print(f"  - The last token's attention reveals what context it uses for prediction")
print(f"  - Attention weights sum to 1.0 (softmax normalized)")
print(f"  - Different heads in the same layer can have very different attention patterns")

