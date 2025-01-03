import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.quantitization import quantitize_dynamic

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


model = quantitize_dynamic(model,dtype=torch.qint8)

model = model.cuda()

model.eval()

def generate_text(prompt, max_length=50):
    inputs = tokenizer.encode(prompt,return_tensors="pt").cuda()
    with torch_no_grad():
        output = model.generate(inputs,max_length=max_length,num_return_sequences=1,no_repeat_ngram_size=2)
    
    generated_text = tokenizer.decode(output[0],skip_special_tokens=True)
    return generated_text

prompt = "What is Clean Code"
gpt_output = generate_text(prompt)
print(f"Generated Text:{gpt_output}")

