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
