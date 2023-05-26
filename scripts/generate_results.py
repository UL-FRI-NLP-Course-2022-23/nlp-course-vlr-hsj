from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel, AutoTokenizer
import nltk
import json
import pandas as pd
from datasets import *
import numpy as np
import torch
from tqdm import tqdm

nltk.download('punkt')
SEED = 42

DEVICE = 'cuda:3'
split = "test"
#model_checkpoint = "vh-student/gpt-sl-oasst1-pairs"
#data_path_pairs = '../../data/processed/prompt_reply_pairs.csv'

model_checkpoint = "vh-student/t5-sl-large-oasst-pairs"
data_path_pairs = 'data/processed/context_reply_pairs.csv'
num_return_sequences = 5


model_name = model_checkpoint.split("/")[-1]


if "t5" in model_name:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to("cuda")
elif "gpt" in model_name:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, truncation_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to("cuda")
else:
    raise ValueError("Model not supported")

print(model.eval())

data = pd.read_csv(data_path_pairs, sep=";")


eval_data = Dataset.from_pandas(data[data['split'] == split][["prompt", "reply"]])

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


eval_data = eval_data.map(convert_to_features, batched=True, load_from_cache_file=False)
eval_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

step = 16 # makeshift batch size - idk how to do this better
indices = np.concatenate([np.arange(0, eval_data.num_rows, step), [eval_data.num_rows]])

outputs = []
model = model.to(DEVICE)
with torch.no_grad():
    with torch.cuda.amp.autocast(True):
        for i in tqdm(range(len(indices[:-1]))):
            generated = model.generate(eval_data[indices[i]:indices[i+1]]['input_ids'].to(DEVICE),
                                    attention_mask=eval_data[indices[i]:indices[i+1]]['attention_mask'].to(DEVICE),
                                    max_new_tokens=128, no_repeat_ngram_size=2, num_beams=5, num_return_sequences=num_return_sequences, early_stopping=True)
            outputs.append(generated)

# results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
results, prediction_lens = [], []
for output in outputs:
    results.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
    prediction_lens.extend([(np.count_nonzero(o.cpu() != tokenizer.pad_token_id)) for o in output])


result_data = pd.read_csv(data_path_pairs, sep=";")

# repeat rows for the number of generated replies
temp = pd.DataFrame(np.repeat(result_data.values, num_return_sequences, axis=0))
temp.columns = result_data.columns
result_data = temp

result_data = result_data[result_data['split'] == split].reset_index(drop=True)
result_data["generated"] = results
result_data["token_prediction_len"] = prediction_lens
result_data.to_csv(f"data/results/prompt_reply_pairs_{num_return_sequences}_generated_{split}_{model_name}.csv", sep=";", index=False)


result_data_1_gen = result_data[::num_return_sequences]
result_data_1_gen.to_csv(f"data/results/prompt_reply_pairs_1_generated_{split}_{model_name}.csv", sep=";", index=False)
