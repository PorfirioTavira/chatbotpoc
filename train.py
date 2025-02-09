import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# 🔧 Check if a GPU with ROCm or CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 📌 Define paths
TXT_FILE_PATH = "output.txt"  # Update with your actual .txt file path
MODEL_NAME = "gpt2"  # Use "gpt2-medium" or "gpt2-large" for larger models

# 🔤 Load Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token, so use EOS instead

# 📖 Load dataset from text file
def load_text_data(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Dataset file {txt_path} not found!")

    dataset = load_dataset("text", data_files={"train": txt_path})
    return dataset

dataset = load_text_data(TXT_FILE_PATH)

# 📝 Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 🎭 Data collator (handles padding dynamically)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 🧠 Load Pretrained GPT-2 Model
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.to(device)  # Move model to GPU or CPU

# ⚙️ Training Configuration
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    eval_strategy="no",  # Evaluate at the end of each epoch
    do_eval= False,
    save_strategy="epoch",  # Save model at the end of each epoch
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=50,
    save_total_limit=2,
    num_train_epochs=10,  # Adjust as needed
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True if torch.cuda.is_available() else False,  # Use mixed precision if GPU is available
    push_to_hub=False,  # Set True if you want to push the model to Hugging Face Hub
)

# 🎯 Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 🚀 Start Training
trainer.train()

# 💾 Save the Fine-Tuned Model
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

print("✅ Training complete! Fine-tuned model saved at './gpt2-finetuned'")
