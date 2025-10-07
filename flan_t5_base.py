import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import torch
import os

from datasets import load_dataset
output_dir="./results"

# LOAD DATA
# ----------------------------
# local path to the CSV
# csv_path = "Financial-QA-10k.csv"
# df = pd.read_csv(csv_path)

# # create HuggingFace dataset
# dataset = Dataset.from_pandas(df)

# # split train/val
# dataset = dataset.train_test_split(test_size=0.1)
# train_dataset = dataset["train"]
# val_dataset = dataset["test"]

dataset = load_dataset("Josephgflowers/Finance-Instruct-500k")

if "train" in dataset and "test" in dataset:
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
else:
    # If the dataset does not have predefined splits, split it manually
    dataset = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]


# ----------------------------
# Model & Tokenizer
# ----------------------------
checkpoint_path = "./results"
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path if os.path.exists(checkpoint_path) else model_name)

max_input_length = 256
max_target_length = 64

# ----------------------------
# Preprocess function
# ----------------------------
def preprocess(examples):
    # inputs = ["question: " + (q if q is not None else "") for q in examples["question"]]
    # inputs = [f"question: {q} context: {ctx} answer:" for q, ctx in zip(examples['question'], examples['context'])]
    inputs = ["user: " + (q if q is not None else "") for q in examples["user"]]

    targets = [(a if a is not None else "") for a in examples["assistant"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )["input_ids"]

    # Replace pad token ids with -100 to ignore in loss
    labels = [[(lid if lid != tokenizer.pad_token_id else -100) for lid in label] for label in labels]

    model_inputs["labels"] = labels
    return model_inputs

# ----------------------------
# Map preprocessing + clean columns
# ----------------------------
train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)

new_output_dir = "./results_finetune_virattt/financial-qa-10K"

# ----------------------------
# Training setup
# ----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=new_output_dir,
    eval_steps=50,
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3, 
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    predict_with_generate=True,
    logging_dir="./logs",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# ----------------------------
# Train
# ----------------------------
# Resume from last checkpoint if available
# last_checkpoint = None
# if os.path.isdir(output_dir):
#     checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
#     if len(checkpoints) > 0:
#     # Sort numerically by step number
#         checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
#         last_checkpoint = checkpoints[-1]
# if last_checkpoint:
#         print(f"Resuming from checkpoint: {last_checkpoint}")
#         trainer.train(resume_from_checkpoint=last_checkpoint)
# else:
#     print("Starting training from scratch...")
#     trainer.train()

# # save final model and tokenizer
# print("Training complete. Saving model to", output_dir)
# trainer.save_model(output_dir)
# tokenizer.save_pretrained(output_dir)

# train - no resume from checkpoint
print("Continuing fine-tuning from previous model weights...")
trainer.train()

# save updated model and tokenizer
trainer.save_model(new_output_dir)
tokenizer.save_pretrained(new_output_dir)
print("New fine-tuned model saved at", new_output_dir)


# ----------------------------
# TEST SAMPLE
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
def generate_answer(question):
    inputs = tokenizer("question: " + question, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# sample_question = "What method is used to record property, plant, and equipment on the financial statements?"
sample_question = "question: What method is used to record property, plant, and equipment on the financial statements? answer:"

print("Answer:", generate_answer(sample_question))