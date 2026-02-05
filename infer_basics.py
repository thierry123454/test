from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Define a prompt
prompt = "Hello, how are you today?"

# Print the prompt string
print("Prompt string:")
print(prompt)
print()

# Tokenize the prompt and get input_ids
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Print input_ids as a list
print("Input IDs list:")
print(input_ids.tolist()[0])
print()

# Decode the input_ids back to text
decoded_text = tokenizer.decode(input_ids[0])
print("Decoded text from IDs:")
print(decoded_text)
print()

# Convert IDs to token strings
token_strings = tokenizer.convert_ids_to_tokens(input_ids[0])
print("Token strings:")
print(token_strings)
print()

# ===== TEXT GENERATION =====
print("=" * 60)
print("TEXT GENERATION COMPARISON")
print("=" * 60)
print()

# Greedy decoding (deterministic)
print("1. GREEDY DECODING (do_sample=False):")
print("-" * 60)
output_greedy = model.generate(
    input_ids,
    max_length=50,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)
generated_text_greedy = tokenizer.decode(output_greedy[0], skip_special_tokens=True)
print(generated_text_greedy)
print()

# Sampling with temperature and top_p (diverse)
print("2. SAMPLING (do_sample=True, temperature=0.7, top_p=0.9):")
print("-" * 60)
output_sampling = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)
generated_text_sampling = tokenizer.decode(output_sampling[0], skip_special_tokens=True)
print(generated_text_sampling)
print()

# Run sampling multiple times to show diversity
print("3. MULTIPLE SAMPLING RUNS (showing diversity):")
print("-" * 60)
for i in range(3):
    output_sample = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output_sample[0], skip_special_tokens=True)
    print(f"Run {i+1}: {generated_text}")
    print()

print("=" * 60)
print("OBSERVATIONS:")
print("- Greedy decoding is deterministic (same output every time)")
print("- Sampling with temperature/top_p produces diverse outputs")
print("- Lower temperature = more focused, higher = more random")
print("- top_p (nucleus sampling) limits to most probable tokens")
print("=" * 60)
