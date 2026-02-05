import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set model to evaluation mode
model.eval()

# ===== PER-STEP GENERATION ANALYSIS =====
print("=" * 80)
print("PER-STEP GENERATION ANALYSIS (Tracking Model Uncertainty)")
print("=" * 80)
print()

# Generate with output_scores to track per-step probabilities
generation_prompt = "The capital of France is"
generation_input_ids = tokenizer.encode(generation_prompt, return_tensors='pt')

print(f"Prompt: '{generation_prompt}'")
print()

# Generate with scores
with torch.no_grad():
    gen_outputs = model.generate(
        generation_input_ids,
        max_length=30,
        do_sample=False,  # Greedy for deterministic analysis
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Extract generated sequence and scores
generated_sequence = gen_outputs.sequences[0]
scores = gen_outputs.scores  # Tuple of tensors, one per generation step

print("STEP-BY-STEP GENERATION:")
print("-" * 80)
print(f"{'Step':<6} {'Token':<20} {'Probability':<15} {'Log Prob':<12} {'Entropy':<12} {'Uncertainty'}")
print("-" * 80)

# Track cumulative text
cumulative_text = generation_prompt

for step_idx, step_scores in enumerate(scores):
    # step_scores shape: (batch_size, vocab_size)
    step_probs = F.softmax(step_scores, dim=-1)[0]  # Get probabilities
    step_log_probs = F.log_softmax(step_scores, dim=-1)[0]  # Get log probabilities
    
    # Get the chosen token (from generated sequence)
    chosen_token_id = generated_sequence[generation_input_ids.shape[1] + step_idx].item()
    chosen_token_str = tokenizer.decode([chosen_token_id])
    chosen_prob = step_probs[chosen_token_id].item()
    chosen_log_prob = step_log_probs[chosen_token_id].item()
    
    # Calculate entropy (measure of uncertainty)
    # H = -sum(p * log(p))
    entropy = -(step_probs * step_log_probs).sum().item()
    
    # Determine uncertainty level
    if chosen_prob > 0.8:
        uncertainty = "Very confident"
    elif chosen_prob > 0.5:
        uncertainty = "Confident"
    elif chosen_prob > 0.3:
        uncertainty = "Uncertain"
    else:
        uncertainty = "Very uncertain"
    
    print(f"{step_idx+1:<6} {repr(chosen_token_str):<20} {chosen_prob:<15.6f} {chosen_log_prob:<12.4f} {entropy:<12.4f} {uncertainty}")
    
    cumulative_text += chosen_token_str

print()
print(f"Final generated text: '{cumulative_text}'")
print()

# Show top-3 alternatives for uncertain steps
print("=" * 80)
print("DETAILED ANALYSIS OF UNCERTAIN STEPS")
print("=" * 80)
print()

for step_idx, step_scores in enumerate(scores):
    step_probs = F.softmax(step_scores, dim=-1)[0]
    step_log_probs = F.log_softmax(step_scores, dim=-1)[0]
    
    chosen_token_id = generated_sequence[generation_input_ids.shape[1] + step_idx].item()
    chosen_prob = step_probs[chosen_token_id].item()
    
    # Only show details for uncertain steps (probability < 0.7)
    if chosen_prob < 0.7:
        print(f"Step {step_idx + 1} - Model was uncertain (chosen token prob: {chosen_prob:.4f})")
        print("Top 5 alternatives:")
        
        top_5_probs, top_5_indices = torch.topk(step_probs, 5)
        for rank, (prob, idx) in enumerate(zip(top_5_probs, top_5_indices), 1):
            token_str = tokenizer.decode([idx.item()])
            log_prob = step_log_probs[idx].item()
            marker = "← CHOSEN" if idx.item() == chosen_token_id else ""
            print(f"  {rank}. {repr(token_str):<20} prob={prob.item():.6f}, logprob={log_prob:.4f} {marker}")
        print()

if all(F.softmax(step_scores, dim=-1)[0][generated_sequence[generation_input_ids.shape[1] + idx].item()].item() >= 0.7 
       for idx, step_scores in enumerate(scores)):
    print("All steps were confident (probability ≥ 0.7)")
    print()
