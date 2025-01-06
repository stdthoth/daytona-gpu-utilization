from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")

# Generate text
input_text = "Who is Gengis Khan"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=300, num_return_sequences=1)
print(tokenizer.decode(output[0], skip_special_tokens=True))


