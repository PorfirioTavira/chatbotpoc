from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")

input_text = "2 + 2 = "
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=10, temperature=.7)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:", generated_text)
