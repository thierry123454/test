import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set model to evaluation mode
model.eval()

# Define a prompt
prompt = "The capital of France is"

print("Prompt:")
print(prompt)
print()

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')

print("Input IDs:")
print(input_ids.tolist()[0])
print()

# Forward pass (no gradient computation needed)
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

print("Logits shape:", logits.shape)
print("(batch_size, sequence_length, vocab_size)")
print()

# Extract next-token logits (logits for the last position)
next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

print("Next-token logits shape:", next_token_logits.shape)
print()

# Compute log probabilities using log_softmax
log_probs = F.log_softmax(next_token_logits, dim=-1)

# Compute probabilities using softmax
probs = F.softmax(next_token_logits, dim=-1)

print("=" * 80)
print("TOP 10 TOKENS WITH PROBABILITIES AND LOG PROBABILITIES")
print("=" * 80)
print()

# Get top 10 tokens
top_k = 10
top_probs, top_indices = torch.topk(probs[0], top_k)
top_log_probs = log_probs[0][top_indices]

print(f"{'Rank':<6} {'Token ID':<10} {'Token':<20} {'Probability':<15} {'Log Prob':<12}")
print("-" * 80)

for i in range(top_k):
    token_id = top_indices[i].item()
    token_str = tokenizer.decode([token_id])
    prob = top_probs[i].item()
    log_prob = top_log_probs[i].item()
    
    print(f"{i+1:<6} {token_id:<10} {repr(token_str):<20} {prob:<15.6f} {log_prob:<12.4f}")

print()
print("=" * 80)
print("GREEDY DECODING CHOICE")
print("=" * 80)
print()

# The token chosen by greedy decoding (argmax)
greedy_token_id = torch.argmax(next_token_logits, dim=-1).item()
greedy_token_str = tokenizer.decode([greedy_token_id])
greedy_prob = probs[0][greedy_token_id].item()
greedy_log_prob = log_probs[0][greedy_token_id].item()

print(f"Greedy token ID:     {greedy_token_id}")
print(f"Greedy token:        {repr(greedy_token_str)}")
print(f"Probability:         {greedy_prob:.6f}")
print(f"Log probability:     {greedy_log_prob:.4f}")
print()

# Verify it's the same as the top-1 token
print(f"Verification: Greedy choice matches top-1 token: {greedy_token_id == top_indices[0].item()}")
print()

# Show the completed text
completed_text = prompt + greedy_token_str
print("Completed text with greedy token:")
print(f"'{completed_text}'")
