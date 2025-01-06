from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("text", data_files={"train": "train.txt"})

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add or set a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as padding token

# Tokenize data
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=50)
    tokens["labels"] = tokens["input_ids"].copy()  # Add labels for computing the loss
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Handle small datasets: Skip split if the dataset is too small
if len(tokenized_dataset["train"]) > 1:
    tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.5)
else:
    # Use the same dataset for both train and test
    tokenized_dataset = {"train": tokenized_dataset["train"], "test": tokenized_dataset["train"]}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=300,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")