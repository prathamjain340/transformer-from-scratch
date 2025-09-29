import os
from datasets import load_dataset, dataset_dict, concatenate_datasets
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import numpy as np
import torch
torch.backends.cudnn.benchmark = True

model_name = "google/flan-t5-large"
output_dir = "flan_t5_multitask_checkpoints"
fp16 = True
seed = 42

# according to rtx 4080
per_device_batch_size = 2
gradient_accumulation_steps = 4         # effective batch size of 8
epochs_sum = 1                          # can increase later
epochs_qa = 2                           # more for QA as SQuAD is smaller, qa benefits from more epochs
learning_rate = 5e-5

# sequence lengths
# summary_max_input_length = 384
summary_max_input_length = 256
summary_max_target_length = 128
# qa_max_input_length = 512
qa_max_input_length = 256
qa_max_target_length = 64

# datasets
summarization_dataset = ("cnn_dailymail", "3.0.0")
qa_dataset = "squad_v2"

# save and logging
save_total_limit = 3
logging_steps = 100
save_steps = 50
eval_steps = 500

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_name=model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ensure tokenizer has an EOS token for padding and seperation:
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": ""})
    tokenizer.pad_token = tokenizer.eos_token # pad with eos token

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # if tokenizer extended vocab (rare), resize embeddings
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def preprocess_summarization(example, tokenizer):
    inputs = ["summarize: " + a.strip().replace("\n", " ") for a in example["article"]]
    model_inputs = tokenizer(inputs, max_length=summary_max_input_length, truncation=True, padding="max_length")

    labels = tokenizer(example["highlights"], max_length=summary_max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]

    # return only tokenizer inputs, no raw text
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": model_inputs["labels"],
    }

def preprocess_qa(example, tokenizer):
    # SQuAD format: question, context, answers -> answers["text"](list)
    inputs = []
    targets = []
    for q, c, a in zip(example["question"], example["context"], example["answers"]):
        # choose first answer if available
        answer_text = a["text"][0] if len(a["text"]) > 0 else ""
        # build input with prefix
        inp = f"question: {q.strip()}  context: {c.strip().replace(chr(10), ' ')}"
        inputs.append(inp)
        targets.append(answer_text if answer_text else "unanswerable")

    model_inputs = tokenizer(inputs, max_length=qa_max_input_length, truncation=True, padding="max_length")
    # with tokenizer.as_target_tokenizer():
    labels = tokenizer(targets, max_length=qa_max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": model_inputs["labels"],
    }

def main():
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.config.use_cache = False  # silence the warnings. may help training

    # put model to device automatically via trainer

    print("Loading datasets...")
    ds_sum = load_dataset(*summarization_dataset)["train"]
    ds_sum_val = load_dataset(*summarization_dataset)["validation"]

    ds_qa = load_dataset(qa_dataset)["train"]
    ds_qa_val = load_dataset(qa_dataset)["validation"]

    # we'll preprocess separately and then concatenate
    print("Tokenizing summarization dataset...")
    sum_tokenized = ds_sum.map(
        lambda ex: preprocess_summarization(ex, tokenizer),
        batched=True,
        remove_columns=ds_sum.column_names,
    )
    sum_tokenized_val = ds_sum_val.map(
        lambda ex: preprocess_summarization(ex, tokenizer),
        batched=True,
        remove_columns=ds_sum_val.column_names,
    )

    print("Tokenizing QA dataset...")
    qa_tokenized = ds_qa.map(
        lambda ex: preprocess_qa(ex, tokenizer),
        batched=True,
        remove_columns=ds_qa.column_names,
    )
    qa_tokenized_val = ds_qa_val.map(
        lambda ex: preprocess_qa(ex, tokenizer),
        batched=True,
        remove_columns=ds_qa_val.column_names,
    )

    # add a column to identify the task so we can optionally sample/mix differently
    def add_task_flag(ex, task_name):
        ex["task"] = [task_name] * len(ex["input_ids"])
        return ex
    
    sum_tokenized = sum_tokenized.map(lambda ex: add_task_flag(ex, "summarization"), batched=True)
    sum_tokenized_val = sum_tokenized_val.map(lambda ex: add_task_flag(ex, "summarization"), batched=True)
    qa_tokenized = qa_tokenized.map(lambda ex: add_task_flag(ex, "qa"), batched=True)
    qa_tokenized_val = qa_tokenized_val.map(lambda ex: add_task_flag(ex, "qa"), batched=True)

    # option: downsample or weight tasks - for now just concatenate
    print("concatenating datasets for multitask training...")
    train_combined = concatenate_datasets([sum_tokenized, qa_tokenized])
    val_combined = concatenate_datasets([sum_tokenized_val, qa_tokenized_val])
    # remove task column as we dont need it for training
    train_combined = train_combined.remove_columns("task")
    val_combined = val_combined.remove_columns("task")


    # shuffle combined dataset
    train_combined = train_combined.shuffle(seed=seed)

    # data collator that will dynamically pad the inputs received, as well as the labels.
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # metrics: wont compute complex metrics here, but can add ROUGE/EM/F1 later
    def compute_metrics(eval_preds):
        # eval_preds is a tuple (predictions, labels)
        # preds, labels are numpy arrays
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # basic metric: average length or exact match fraction - placeholder
        result = {"pred_len": np.mean([len(p.split()) for p in decoded_preds])}
        return result

        # simple exact match metric: how many predictions match labels exactly
        exact_matches = sum([pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)])
        total = len(decoded_labels)
        exact_match_rate = exact_matches / total if total > 0 else 0.0

        return {
            "exact_match": exact_match_rate,
            "total": total,
        }
    
    # training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs_sum + epochs_qa,  # total epochs
        learning_rate=learning_rate,
        fp16=fp16,
        # evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to=[], # set to ["tensorboard"] or ["wandb"] to enable logging
        save_strategy="steps",
        remove_unused_columns=False, # important as we used custom columns
        seed=seed,
        dataloader_num_workers=8,
    )

    print("Initializing Trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_combined,
        eval_dataset=val_combined,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Resume from last checkpoint if available
    last_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if len(checkpoints) > 0:
        # Sort numerically by step number
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            last_checkpoint = checkpoints[-1]

    # Start training
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Starting training from scratch...")
        trainer.train()

    print("Training complete. Saving model to", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # save final model and tokenizer
    print("Training complete. Saving model to", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # quick interactive test
    def generate_sample(input_text, max_new_tokens=64):
        inputs = tokenizer(input_text, return_tensors="pt").to(trainer.model.device)
        outs = trainer.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
        return tokenizer.decode(outs[0], skip_special_tokens=True)
    
    # sample generations (summarization)
    sample_article = "summarize: the quick brown fox jumps over the lazy dog. " * 30
    print("sample summary:", generate_sample(sample_article, max_new_tokens=100))

    # sample generations (QA)
    sample_qa = "question: where does the fox jump?  context: the quick brown fox jumps over the lazy dog."
    print("sample answer:", generate_sample(sample_qa, max_new_tokens=32))

if __name__ == "__main__":
    main()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~