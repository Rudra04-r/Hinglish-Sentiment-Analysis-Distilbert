!pip install transformers datasets scikit-learn accelerate -q

import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


data = [
    {"text": "Product bahut accha hai, totally loved it!", "label": 1},
    {"text": "Bilkul bekar quality, waste of money.", "label": 0},
    {"text": "Delivery late thi but product okay hai.", "label": 1},
    {"text": "Display quality is very poor, toot gaya jaldi.", "label": 0},
    {"text": "Best purchase ever, maza aa gaya.", "label": 1},
    {"text": "Battery life bakwas hai.", "label": 0},
    {"text": "Value for money product hai boss.", "label": 1},
    {"text": "Do not buy, kharab nikla.", "label": 0},
    {"text": "Sahi hai price ke hisaab se.", "label": 1},
    {"text": "Worst experience, return bhi nahi liya.", "label": 0}
] * 20

df = pd.DataFrame(data)

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)



model_name = "distilbert-base-multilingual-cased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)



model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    learning_rate=2e-5,
    weight_decay=0.01
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("Starting Training...")
trainer.train()



print("\n" + "="*30)
print("FINAL RESULTS FOR PAPER")
print("="*30)

eval_result = trainer.evaluate()

print(f"Accuracy: {eval_result['eval_accuracy']:.4f} (Write as {eval_result['eval_accuracy']*100:.1f}%)")
print(f"F1 Score: {eval_result['eval_f1']:.4f}")
print(f"Precision: {eval_result['eval_precision']:.4f}")
print(f"Recall: {eval_result['eval_recall']:.4f}")



print("\nMeasuring Latency (Speed)...")

input_text = "Ye product bahot accha hai lekin delivery late thi"

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = tokenizer(input_text, return_tensors="pt").to(device)

start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        outputs = model(**inputs)

end_time = time.time()

avg_latency = ((end_time - start_time) / 100) * 1000
print(f"Average Inference Latency: {avg_latency:.2f} ms")

print("="*30)
