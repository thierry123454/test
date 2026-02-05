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
print()

# ===== CONTINUATION SCORING =====
print("=" * 80)
print("CONTINUATION SCORING")
print("=" * 80)
print()

def score_continuation(prompt, continuation):
    """
    Compute the total log probability of a continuation given a prompt.
    
    Args:
        prompt: The conditioning text (string)
        continuation: The text to score (string)
    
    Returns:
        float: Total log probability of the continuation
    """
    # Tokenize prompt and full text (prompt + continuation)
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
    full_text = prompt + continuation
    full_ids = tokenizer.encode(full_text, return_tensors='pt')
    
    # Get the continuation tokens (everything after the prompt)
    continuation_length = full_ids.shape[1] - prompt_ids.shape[1]
    
    if continuation_length <= 0:
        return 0.0  # No continuation to score
    
    # Forward pass on the full sequence
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # Shape: (1, seq_len, vocab_size)
    
    # Compute log probabilities for each position
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: (1, seq_len, vocab_size)
    
    # Sum log probabilities for continuation tokens
    total_log_prob = 0.0
    
    # For each token in the continuation, get its log probability
    # Note: logits[0, i] predicts token at position i+1
    for i in range(continuation_length):
        position = prompt_ids.shape[1] + i - 1  # Position in logits that predicts this token
        token_id = full_ids[0, prompt_ids.shape[1] + i].item()  # The actual token
        token_log_prob = log_probs[0, position, token_id].item()
        total_log_prob += token_log_prob
    
    return total_log_prob


# Example: Compare two continuations
test_prompt = "The capital of France is"
continuation1 = " Paris"
continuation2 = " London"

print(f"Prompt: '{test_prompt}'")
print()

score1 = score_continuation(test_prompt, continuation1)
score2 = score_continuation(test_prompt, continuation2)

print(f"Continuation 1: '{continuation1}'")
print(f"  Total log probability: {score1:.4f}")
print(f"  Perplexity: {torch.exp(torch.tensor(-score1 / len(tokenizer.encode(continuation1)))).item():.4f}")
print()

print(f"Continuation 2: '{continuation2}'")
print(f"  Total log probability: {score2:.4f}")
print(f"  Perplexity: {torch.exp(torch.tensor(-score2 / len(tokenizer.encode(continuation2)))).item():.4f}")
print()

if score1 > score2:
    print(f"✓ Model prefers: '{continuation1}' (higher log probability)")
    print(f"  Difference: {score1 - score2:.4f} log probability units")
elif score2 > score1:
    print(f"✓ Model prefers: '{continuation2}' (higher log probability)")
    print(f"  Difference: {score2 - score1:.4f} log probability units")
else:
    print("Model is indifferent (equal log probabilities)")

print()
print("=" * 80)
print("ADDITIONAL EXAMPLES")
print("=" * 80)
print()

# More examples
examples = [
    ("The sky is", [" blue", " green", " purple"]),
    ("I love to eat", [" pizza", " rocks", " delicious food"]),
    ("The president of the United States is", [" elected", " appointed", " chosen by lottery"]),
]

for prompt_ex, continuations in examples:
    print(f"Prompt: '{prompt_ex}'")
    scores = []
    for cont in continuations:
        score = score_continuation(prompt_ex, cont)
        scores.append((cont, score))
        print(f"  '{cont}': {score:.4f}")
    
    # Find the best continuation
    best_cont, best_score = max(scores, key=lambda x: x[1])
    print(f"  → Model prefers: '{best_cont}'")
    print()
