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
print("COSINE SIMILARITY BETWEEN TOKEN PAIRS ACROSS LAYERS")
print("=" * 80)
print()

print(f"Prompt: '{prompt}'")
print()

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

print(f"Tokens: {tokens}")
print()

# Find positions of "France" and "Paris"
france_pos = None
paris_pos = None

for i, token in enumerate(tokens):
    if 'France' in token:
        france_pos = i
    if 'Paris' in token:
        paris_pos = i

print(f"Token positions:")
print(f"  'France' at position {france_pos}: '{tokens[france_pos]}'")
print(f"  'Paris' at position {paris_pos}: '{tokens[paris_pos]}'")
print()

# Forward pass with output_hidden_states=True
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states

num_layers = len(hidden_states)

print("=" * 80)
print("COSINE SIMILARITY: 'France' vs 'Paris' Across Layers")
print("=" * 80)
print()

print(f"{'Layer':<15} {'Cosine Similarity':<20} {'Interpretation'}")
print("-" * 80)

similarities = []

for l in range(num_layers):
    # Get hidden states for France and Paris at this layer
    france_vec = hidden_states[l][0, france_pos, :]
    paris_vec = hidden_states[l][0, paris_pos, :]
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(france_vec.unsqueeze(0), paris_vec.unsqueeze(0)).item()
    similarities.append(cos_sim)
    
    # Interpretation
    if cos_sim > 0.9:
        interp = "Very similar"
    elif cos_sim > 0.7:
        interp = "Similar"
    elif cos_sim > 0.5:
        interp = "Moderately similar"
    elif cos_sim > 0.3:
        interp = "Somewhat different"
    else:
        interp = "Very different"
    
    layer_name = "Embedding" if l == 0 else f"Layer {l}"
    print(f"{layer_name:<15} {cos_sim:<20.6f} {interp}")

print()

# Find layer with maximum and minimum similarity
max_sim_layer = similarities.index(max(similarities))
min_sim_layer = similarities.index(min(similarities))

max_layer_name = "Embedding" if max_sim_layer == 0 else f"Layer {max_sim_layer}"
min_layer_name = "Embedding" if min_sim_layer == 0 else f"Layer {min_sim_layer}"

print(f"Maximum similarity: {max(similarities):.6f} at {max_layer_name}")
print(f"Minimum similarity: {min(similarities):.6f} at {min_layer_name}")
print()

# ===== ADDITIONAL ANALYSIS: Germany vs Paris =====
print("=" * 80)
print("ADDITIONAL COMPARISON: 'Germany' vs 'Paris'")
print("=" * 80)
print()

# Find Germany position
germany_pos = None
for i, token in enumerate(tokens):
    if 'Germany' in token:
        germany_pos = i

print(f"Comparing 'Germany' (pos {germany_pos}) vs 'Paris' (pos {paris_pos})")
print()

print(f"{'Layer':<15} {'Cosine Similarity':<20}")
print("-" * 80)

germany_paris_sims = []

for l in range(num_layers):
    germany_vec = hidden_states[l][0, germany_pos, :]
    paris_vec = hidden_states[l][0, paris_pos, :]
    
    cos_sim = F.cosine_similarity(germany_vec.unsqueeze(0), paris_vec.unsqueeze(0)).item()
    germany_paris_sims.append(cos_sim)
    
    layer_name = "Embedding" if l == 0 else f"Layer {l}"
    print(f"{layer_name:<15} {cos_sim:<20.6f}")

print()

# ===== COMPARISON: France vs Germany =====
print("=" * 80)
print("ADDITIONAL COMPARISON: 'France' vs 'Germany'")
print("=" * 80)
print()

print(f"Comparing 'France' (pos {france_pos}) vs 'Germany' (pos {germany_pos})")
print()

print(f"{'Layer':<15} {'Cosine Similarity':<20}")
print("-" * 80)

france_germany_sims = []

for l in range(num_layers):
    france_vec = hidden_states[l][0, france_pos, :]
    germany_vec = hidden_states[l][0, germany_pos, :]
    
    cos_sim = F.cosine_similarity(france_vec.unsqueeze(0), germany_vec.unsqueeze(0)).item()
    france_germany_sims.append(cos_sim)
    
    layer_name = "Embedding" if l == 0 else f"Layer {l}"
    print(f"{layer_name:<15} {cos_sim:<20.6f}")

print()

# ===== SUMMARY =====
print("=" * 80)
print("SUMMARY OF SEMANTIC RELATIONSHIPS")
print("=" * 80)
print()

print("Average cosine similarity across all layers:")
print(f"  France ↔ Paris:   {sum(similarities) / len(similarities):.6f}")
print(f"  Germany ↔ Paris:  {sum(germany_paris_sims) / len(germany_paris_sims):.6f}")
print(f"  France ↔ Germany: {sum(france_germany_sims) / len(france_germany_sims):.6f}")
print()

print("Interpretation:")
print("  - France and Paris should be semantically related (country-capital)")
print("  - Germany and Paris are less related (different country, different capital)")
print("  - France and Germany are related (both countries)")
print()

# Show how similarity evolves
print("Evolution of France-Paris similarity:")
print(f"  Embedding layer:  {similarities[0]:.6f}")
print(f"  Early (Layer 3):  {similarities[3]:.6f}")
print(f"  Middle (Layer 6): {similarities[6]:.6f}")
print(f"  Late (Layer 9):   {similarities[9]:.6f}")
print(f"  Final (Layer 12): {similarities[12]:.6f}")
print()

if similarities[12] > similarities[0]:
    print("→ Similarity INCREASED through the network")
    print("  The model learned to associate France and Paris more strongly")
else:
    print("→ Similarity DECREASED through the network")
    print("  The model differentiated France and Paris representations")
