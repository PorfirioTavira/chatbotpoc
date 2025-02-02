import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Model Parameters
MAX_LENGTH = 50 
TEMPERATURE = 0.1

# Load pretrained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Input text
input_text = "Hello, how are you today?"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate text
output = model.generate(input_ids, max_length=MAX_LENGTH, num_return_sequences=1, temperature=TEMPERATURE, do_sample=True)

# Decode the generated tokens into text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print results
print("Input Text:", input_text)
print("Generated Text:", generated_text)
