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
print("HIDDEN STATES ANALYSIS")
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

# Forward pass with output_hidden_states=True
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    logits = outputs.logits
    hidden_states = outputs.hidden_states

print("=" * 80)
print("HIDDEN STATES STRUCTURE")
print("=" * 80)
print()

# Number of layers (includes embedding layer + all transformer layers)
num_layers = len(hidden_states)
print(f"Number of hidden state layers returned: {num_layers}")
print(f"  - Layer 0: Embedding layer")
print(f"  - Layers 1-{num_layers-1}: Transformer layers")
print()

# Print shape of each hidden state
print("Shape of hidden_states[l] for each layer:")
print("-" * 80)
print(f"{'Layer':<10} {'Shape':<30} {'Description'}")
print("-" * 80)

for l in range(num_layers):
    shape = hidden_states[l].shape
    if l == 0:
        desc = "Embedding layer output"
    else:
        desc = f"Transformer layer {l} output"
    print(f"{l:<10} {str(tuple(shape)):<30} {desc}")

print()
print(f"Shape format: (batch_size, sequence_length, hidden_size)")
print(f"Hidden size (model dimension): {hidden_states[0].shape[-1]}")
print()

# ===== VECTOR NORM ANALYSIS =====
print("=" * 80)
print("VECTOR NORM ANALYSIS ACROSS LAYERS")
print("=" * 80)
print()

# Choose a token position to analyze (let's analyze the last token)
chosen_position = len(tokens) - 1
chosen_token = tokens[chosen_position]

print(f"Analyzing token at position {chosen_position}: '{chosen_token}'")
print()

# Compute L2 norm for this token across all layers
norms = []
for l in range(num_layers):
    # Get the hidden state for the chosen position
    hidden_vec = hidden_states[l][0, chosen_position, :]  # Shape: (hidden_size,)
    
    # Compute L2 norm
    norm = torch.norm(hidden_vec, p=2).item()
    norms.append(norm)

print("L2 Norm of hidden state vector across layers:")
print("-" * 80)
print(f"{'Layer':<10} {'L2 Norm':<15} {'Change from prev':<20}")
print("-" * 80)

for l in range(num_layers):
    if l == 0:
        change = "N/A (first layer)"
    else:
        change = f"{norms[l] - norms[l-1]:+.4f}"
    
    layer_name = "Embedding" if l == 0 else f"Layer {l}"
    print(f"{layer_name:<10} {norms[l]:<15.4f} {change:<20}")

print()

# ===== ANALYZE MULTIPLE POSITIONS =====
print("=" * 80)
print("VECTOR NORMS FOR ALL TOKEN POSITIONS (Last Layer)")
print("=" * 80)
print()

last_layer_idx = num_layers - 1
print(f"Analyzing layer {last_layer_idx} (final transformer layer)")
print()

print(f"{'Position':<10} {'Token':<20} {'L2 Norm':<15}")
print("-" * 80)

for pos in range(len(tokens)):
    token = tokens[pos]
    hidden_vec = hidden_states[last_layer_idx][0, pos, :]
    norm = torch.norm(hidden_vec, p=2).item()
    print(f"{pos:<10} {repr(token):<20} {norm:<15.4f}")

print()

# ===== LAYER-WISE NORM STATISTICS =====
print("=" * 80)
print("LAYER-WISE STATISTICS (All Positions)")
print("=" * 80)
print()

print(f"{'Layer':<10} {'Mean Norm':<15} {'Std Norm':<15} {'Min Norm':<15} {'Max Norm':<15}")
print("-" * 80)

for l in range(num_layers):
    # Compute norms for all positions in this layer
    layer_norms = []
    for pos in range(len(tokens)):
        hidden_vec = hidden_states[l][0, pos, :]
        norm = torch.norm(hidden_vec, p=2).item()
        layer_norms.append(norm)
    
    mean_norm = sum(layer_norms) / len(layer_norms)
    std_norm = (sum((x - mean_norm) ** 2 for x in layer_norms) / len(layer_norms)) ** 0.5
    min_norm = min(layer_norms)
    max_norm = max(layer_norms)
    
    layer_name = "Embedding" if l == 0 else f"Layer {l}"
    print(f"{layer_name:<10} {mean_norm:<15.4f} {std_norm:<15.4f} {min_norm:<15.4f} {max_norm:<15.4f}")

print()

# ===== COSINE SIMILARITY BETWEEN LAYERS =====
print("=" * 80)
print("COSINE SIMILARITY BETWEEN CONSECUTIVE LAYERS")
print("=" * 80)
print()

print(f"Analyzing token at position {chosen_position}: '{chosen_token}'")
print()

print(f"{'Layers':<20} {'Cosine Similarity':<20}")
print("-" * 80)

for l in range(1, num_layers):
    vec_prev = hidden_states[l-1][0, chosen_position, :]
    vec_curr = hidden_states[l][0, chosen_position, :]
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(vec_prev.unsqueeze(0), vec_curr.unsqueeze(0)).item()
    
    layer_prev = "Embedding" if l-1 == 0 else f"Layer {l-1}"
    layer_curr = f"Layer {l}"
    
    print(f"{layer_prev} → {layer_curr:<10} {cos_sim:<20.6f}")

print()
print("Note: Cosine similarity close to 1.0 means vectors point in similar directions")
print("      Lower values indicate more transformation between layers")
print()

# ===== VISUALIZATION DATA =====
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print(f"Model: GPT-2")
print(f"Number of transformer layers: {num_layers - 1}")
print(f"Hidden dimension: {hidden_states[0].shape[-1]}")
print(f"Sequence length: {len(tokens)}")
print()

print("Key Observations:")
print(f"  - Hidden states are available for {num_layers} layers (embedding + {num_layers-1} transformer layers)")
print(f"  - Each hidden state has shape (batch_size, seq_len, hidden_dim)")
print(f"  - Vector norms can be tracked across layers to see how representations evolve")
print(f"  - Cosine similarity shows how much each layer transforms the representation")
