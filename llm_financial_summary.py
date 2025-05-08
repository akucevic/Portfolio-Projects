# Fine-tuning a Domain-Specific LLM for Financial Analysis

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, DatasetDict
import torch

# Load dataset (10-K filings and financial summaries)
dataset = load_dataset("json", data_files={"train": "10k_train.json", "validation": "10k_val.json"})

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-large")

# Tokenization
def tokenize_data(example):
    inputs = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(example["summary"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_datasets = dataset.map(tokenize_data, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./llm_financial",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
