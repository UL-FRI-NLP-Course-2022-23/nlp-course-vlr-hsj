import os

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoConfig

import pandas as pd
import datasets as ds

import nltk
import numpy as np


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB__SERVICE_WAIT"] = 300


data = pd.read_csv("data/processed/context_reply_pairs.csv", sep=";")

train_data = data[data["split"] == "train"]
val_data = data[data["split"] == "val"]
test_data = data[data["split"] == "test"]

train_data = ds.Dataset.from_pandas(train_data[['prompt', 'reply']])
val_data = ds.Dataset.from_pandas(val_data[['prompt', 'reply']])
test_data = ds.Dataset.from_pandas(test_data[['prompt', 'reply']])

tokenizer = AutoTokenizer.from_pretrained("cjvt/gpt-sl-base", truncation=True, truncation_side='left')

model = AutoModelForCausalLM.from_pretrained("cjvt/gpt-sl-base")
print('Model loaded')

def convert_to_features(examples):
    # prefix_in = "Uporabnik: "
    prefix_in = ""
    examples["prompt"] = [prefix_in + prompt for prompt in examples["prompt"]]
    # prefix_out = "Asistent: "
    prefix_out = ""
    examples["reply"] = [prefix_out + reply for reply in examples["reply"]]
    
    model_inputs = tokenizer(examples['prompt'], pad_to_max_length=True, max_length=512, truncation=True, return_tensors='pt')

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['reply'], pad_to_max_length=True, max_length=128, truncation=True, return_tensors='pt')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_data = train_data.map(convert_to_features, batched=True, load_from_cache_file=False)
val_data = val_data.map(convert_to_features, batched=True, load_from_cache_file=False)
test_data = test_data.map(convert_to_features, batched=True, load_from_cache_file=False)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = ds.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

training_args = TrainingArguments(
    output_dir="models/gpt-ft-3-context", #The output directory
    evaluation_strategy='no',
    #evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    push_to_hub=False, 
    fp16=True,
    load_best_model_at_end=False,
    #load_best_model_at_end=True, # load best model at end so we save the best model instead of the last model
    report_to='wandb',
    eval_accumulation_steps = 1,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    #eval_dataset=val_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model()