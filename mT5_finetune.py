from huggingface_hub import notebook_login
from transformers import Seq2SeqTrainingArguments
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from evaluate import load
import evaluate
from transformers import DataCollatorForSeq2Seq
import json
from transformers import Seq2SeqTrainer

EXP_LEN = 0.7

rouge_score = evaluate.load('rouge')

nltk.download("punkt")

notebook_login()

model_checkpoint = "google/mt5-small"

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    global EXP_LEN
    max_input_length = 512
    max_target_length = int(max_input_length*EXP_LEN)
    model_inputs = tokenizer(
        examples["orig"], max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["summary"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


train_data = None
with open(f'./data/train_data_summarized_{EXP_LEN*100}.json') as f:
    train_data = json.load(f)

train_examples = {'orig': [], 'summary': []}

for item in train_data:
    train_examples['orig'].append(item['orig'])
    train_examples['summary'].append(item['summary'])

val_data = None
with open(f'./data/val_data_summarized_{EXP_LEN*100}.json') as f:
    val_data = json.load(f)

val_examples = {'orig': [], 'summary': []}

for item in val_data:
    val_examples['orig'].append(item['orig'])
    val_examples['summary'].append(item['summary'])

batch_size = 8
num_train_epochs = 8
logging_steps = len(train_data) // batch_size
model_name = model_checkpoint.split("/")[-1]
tokenized_datasets = {'train': preprocess_function(train_examples), 'validation': preprocess_function(val_examples)}

args = Seq2SeqTrainingArguments(
    output_dir=f"./models/{model_name}-finetuned-squad-{EXP_LEN*100}",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=False,
)


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
