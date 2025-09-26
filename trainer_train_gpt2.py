import os
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "gpt2"                 # base model to load
output_dir = "hf_checkpoints"       # where checkpoints and final model will be saved
max_length = 256                    # max total length for (prompt + summary)
max_input_length = 192              # max length to allow for the input/prompt portion
batch_size = 4
eval_batch_size = 4
epochs = 2
learning_rate = 5e-5
save_strategy = "steps"             # or "steps"
save_steps = 1000                   # used if save_strategy == "steps"
gen_max_new_tokens = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_name=model_name):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    # ensure tokenizer has an EOS token for padding and seperation:
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": ""})
    tokenizer.pad_token = tokenizer.eos_token # pad with eos token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    # if tokenizer extended vocab (rare), resize embeddings
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def preprocess_examples(example: Dict, tokenizer: GPT2TokenizerFast, max_length=max_length, max_input_len=max_input_length):
    article = example.get("article") or example.get("input") or example.get("text") or ""
    summary = example.get("highlights") or example.get("output") or example.get("summary") or ""
    prompt_text = build_prompt(article)

    # encode prompt (without special tokens) to get prompt length (untruncated)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    # ensure prompt is not longer than allowed
    if len(prompt_ids) > max_input_len:
        prompt_ids = prompt_ids[-max_input_len] # keep last tokens of prompt (or could truncate from front)

    # encode combined sequence with truncations to max_length
    # combine prompt (can be truncated) + eos + summary
    combined_text = tokenizer.decode(prompt_ids, clean_up_tokenization_spaces=True) + tokenizer.eos_token + summary
    # pad here as trainer collator expects tensors
    enc = tokenizer(combined_text, truncation=True, max_length=max_length, padding="max_length", return_attention_mask=True)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # build labels: copy of input_ids, but set prompt tokens to -100
    # find the prompt token length in the *truncated/padded* input:
    # re-encode prompt within max_length to know how many prompt tokens survived
    prompt_enc = tokenizer(tokenizer.decode(prompt_ids, clean_up_tokenization_spaces=True), truncation=True, max_length=max_length, add_special_tokens=False)

    prompt_len = len(prompt_enc["input_ids"])

    labels = input_ids.copy()
    for i in range(prompt_len):
        labels[i] = -100 # ignore prompt positions in loss

    # convert to dict of lists
    return {
        "input_ids": input_ids,
        "attention_mask" : attention_mask,
        "labels" : labels
    }

def preprocess_for_bart(example, tokenizer: GPT2TokenizerFast):
    # encode input (article)
    inputs = tokenizer(example["article"], truncation=True, max_length=max_input_length, padding="max_length")
    # encode target (summary)
    targets = tokenizer(example["highlights"], truncation=True, max_length=max_length, padding="max_length")
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"],
    }

def collate_remove_unused_columns(features):
    # trainer will convert lists -> tensors automatically, we return as-is
    return features

def build_prompt(article: str):
    # simple prompt prefix (you can change wording)
    return "Summarize: " + article.strip()

def generate_summary(model, tokenizer, prompt: str, device, max_new_tokens=gen_max_new_tokens, temperature=0.9, top_k=50, top_p=0.95, repetition_penalty=1.2):
    model.eval()
    input_prompt = build_prompt(prompt)
    # input_prompt = """summarize: (CNN) -- Hillary Clinton, because she's the Democrat's presumptive 2016 front-runner, 
    # has become the target du jour. The latest craziness comes from Matt Drudge, who publishes the Drudge Report, 
    # the political equivalent of an online supermarket tabloid. Drudge funnels anonymous propaganda into the mainstream media."""

    inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
    # generate using HF generate (handles past_key_values etc.)
    out_ids = model.generate(input_ids=inputs["input_ids"], max_new_tokens=max_new_tokens, attention_mask=inputs["attention_mask"], do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    # decode and remove the prompt prefix from the generated text
    generated_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
    # strip leading prompt if model reproduced it
    if generated_text.startswith(input_prompt):
        return generated_text[len(input_prompt):].strip()
    return generated_text

def main():
    os.makedirs(output_dir, exist_ok=True)
    tokenizer, model = load_model_and_tokenizer(model_name)
    # model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model.to(device)

    # load dataset (CNN/DM). can change to any dataset with keys 'article' and 'highlights'
    raw = load_dataset("cnn_dailymail", "3.0.0")

    # train on a subset to debug
    train_ds = raw["train"]
    val_ds = raw["validation"]

    # map preprocess (this will produce tokenized inputs/labels)
    def _map_fn(ex):
        return preprocess_examples(ex, tokenizer, max_length=max_length, max_input_len=max_input_length)
        # return preprocess_for_bart(ex, tokenizer)
    
    print("Tokenizing train split...")
    tokenized_train = train_ds.map(_map_fn, remove_columns=train_ds.column_names, batched=False)
    print("Tokenizing validation split...")
    tokenized_val = val_ds.map(_map_fn, remove_columns=val_ds.column_names, batched=False)

    # training arguments
    per_device_train_batch_size = batch_size
    per_device_eval_batch_size = eval_batch_size
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=False,
        report_to=[]
    )

    # data collator for causal lm (trainer will use labels we prepared)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # traier
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator
    )

    # resume from a checkpoint if present in output_dir
    last_ckpt = None
    #HF saves with directories like output_dir/checkpoint-xxxx or checkpoint-epoch
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint"):
            candidate = os.path.join(output_dir, name)
            if os.path.isdir(candidate):
                last_ckpt = candidate
    if last_ckpt:
        print(f"resuming from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("no checkpoint found. Starting training from model weights loaded above")
        trainer.train()
    
    # save final model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # quick test generation
    test_prompt = "the cat sat on the mat"
    print("\n===sample generation after training===")
    print(generate_summary(model, tokenizer, prompt=test_prompt, device=device))

if __name__ == "__main__":
    main()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~