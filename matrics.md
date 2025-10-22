# Fine-tuning T5 for Korean Hangul → Korean Braille Translation

This repository demonstrates how to fine-tune a **T5 model** for translating Korean text (Hangul) into **Korean Braille**.  
It also explains the evaluation metrics commonly used for NLP tasks, such as translation, classification, and semantic understanding.

---

## 📘 Task Overview

The goal is to train a **T5-based sequence-to-sequence model** that can convert Korean text (Hangul) into Korean Braille.  
Braille translation is a structured language generation task that requires both **linguistic accuracy** and **structural precision**, making metrics like BLEU and F1 crucial for evaluation.

---

## 📊 Evaluation Metrics

Below are the common metrics used in Korean NLP benchmarks and what they measure:

| Metric | Meaning | Common Use Case | Description |
|---------|----------|----------------|--------------|
| **ynat (macro F1)** | News Classification | Text Classification | Macro F1 calculates the average F1 score per class, treating all classes equally regardless of size. |
| **sts (pearsonr / F1)** | Semantic Textual Similarity | Sentence Pair Tasks | Measures correlation (Pearson) between predicted and actual similarity scores. F1 may be used when labels are categorical. |
| **nli (acc)** | Natural Language Inference | Entailment / Contradiction | Accuracy measures the ratio of correct predictions for sentence-pair classification. |
| **ner (entity-level F1)** | Named Entity Recognition | Sequence Labeling | Entity-level F1 checks how well entities (like names, dates) are identified. |
| **re (micro F1)** | Relation Extraction | Semantic Relation Detection | Micro F1 aggregates contributions from all classes to calculate global precision and recall. |
| **dp (LAS)** | Dependency Parsing | Syntax Parsing | Labeled Attachment Score (LAS) measures how well the model predicts both the correct head and dependency label. |
| **mrc (EM / F1)** | Machine Reading Comprehension | QA Tasks | EM (Exact Match) checks if the answer exactly matches the gold answer; F1 measures token-level overlap. |
| **BLEU score** | Translation Quality | Machine Translation | Compares n-gram overlap between generated and reference text. Commonly used for translation tasks like Hangul → Braille. |

> 🧠 For **translation**, the most relevant metric is **BLEU**, but F1 and EM can also be tracked for strict correctness evaluation.

---

## 🧩 Example: Fine-tuning T5 for Korean Braille Translation

Below is a simple example using the **Hugging Face Transformers** library to fine-tune a T5 model.

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

# Load tokenizer and model (e.g., pretrained T5-base or Korean T5)
tokenizer = T5Tokenizer.from_pretrained("paust/pko-t5-base")
model = T5ForConditionalGeneration.from_pretrained("paust/pko-t5-base")

# Load your dataset
# Example: a dataset with columns ["hangul", "braille"]
dataset = load_dataset("csv", data_files={"train": "train.csv", "validation": "valid.csv"})

# Preprocess function
def preprocess_function(examples):
    inputs = ["translate Hangul to Braille: " + text for text in examples["hangul"]]
    targets = [braille for braille in examples["braille"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir="./logs",
)

# Define BLEU metric
bleu = load_metric("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu_score = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    return {"bleu": bleu_score["score"]}

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start fine-tuning
trainer.train()
