from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# Load dataset
dataset = load_dataset('conll2012_ontonotesv5', 'english_v4')

# Load tokenizer and model
model_name = "nielsr/coref-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=8)

print(dataset['train'].column_names)


# Add a new column 'labels' with default values
dataset['train'] = dataset['train'].add_column('labels', [0] * len(dataset['train']))

def preprocess_data(examples):
    sentences = examples['sentences']
    
    tokenized_inputs = {'input_ids': [], 'attention_mask': [], 'labels': []}
    max_length = 0
    for sentence in sentences:
        words = [word.get('form', '') for word in sentence]
        tokenized_input = tokenizer(' '.join(words), truncation=True)
        max_length = max(max_length, len(tokenized_input['input_ids']))
    
    for sentence in sentences:
        words = [word.get('form', '') for word in sentence]
        tokenized_input = tokenizer(' '.join(words), truncation=True, max_length=max_length, padding='max_length')
        tokenized_inputs['input_ids'].append(tokenized_input['input_ids'])
        tokenized_inputs['attention_mask'].append(tokenized_input['attention_mask'])
        tokenized_inputs['labels'].append([0] * max_length)  # Assuming 0 as the default label
    return tokenized_inputs

tokenized_datasets = dataset.map(preprocess_data, batched=True)

# Define metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[label for label in labels[i] if label != -100] for i in range(len(labels))]
    true_predictions = [
        [pred for (pred, label) in zip(prediction, labels[i]) if label != -100]
        for i, prediction in enumerate(predictions)
    ]
    results = precision_recall_fscore_support(np.concatenate(true_labels), np.concatenate(true_predictions), average='weighted')
    return {
        'precision': results[0],
        'recall': results[1],
        'f1': results[2],
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
eval_results = trainer.evaluate()
print(eval_results)