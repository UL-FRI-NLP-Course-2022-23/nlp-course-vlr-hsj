
import torch
import nltk
import json
import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

SEED = 42
nltk.download('punkt')

model_path = "/d/hpc/projects/FRI/vh0153/nlp-course-vlr-hsj/models"


model_checkpoint = "cjvt/t5-sl-large"
name = model_checkpoint.split("/")[-1]

epochs = 40
batch_size = 64

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# LOAD data
data = pd.read_csv("/d/hpc/projects/FRI/vh0153/nlp-course-vlr-hsj/data/processed/prompt_reply_pairs.csv", sep=";")
data

train_data = data[data["split"] == "train"]
val_data = data[data["split"] == "val"]
test_data = data[data["split"] == "test"]


train_data = Dataset.from_pandas(train_data[['prompt', 'reply']])
val_data = Dataset.from_pandas(val_data[['prompt', 'reply']])
test_data = Dataset.from_pandas(test_data[['prompt', 'reply']])

def convert_to_features(examples):
    prefix_in = "Uporabnik: "
    # prefix_in = ""
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

metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}



# training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"{model_path}/{name}-finetuned-assistant",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=epochs,
    predict_with_generate=True,
    fp16=False, # setting this to true gives loss 0.0 at every step for some reason
    push_to_hub=False, 
    load_best_model_at_end=True, # load best model at end so we save the best model instead of the last 
    report_to='wandb'
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()
trainer.save_model(f"{model_path}/{name}-finetuned-assistant")



val_results = trainer.evaluate()
print('Val results: ', val_results)


test_results = trainer.predict(test_dataset=test_data)
print('Test results:', test_results.metrics)



input = "Uporabnik: Kdo je France Prešeren?"
input = tokenizer(input, return_tensors="pt").to("cuda")
outputs = trainer.model.generate(**input, max_length=128, no_repeat_ngram_size=2, num_beams=5, num_return_sequences=5)
tokenizer.decode(outputs[0], skip_special_tokens=True)
tokenizer.batch_decode(outputs, skip_special_tokens=True)