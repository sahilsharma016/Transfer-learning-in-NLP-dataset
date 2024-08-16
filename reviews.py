# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# from sklearn.metrics import precision_recall_fscore_support
# import numpy as np

# # Load your dataset from a CSV file
# dataset = load_dataset('csv', data_files='Reviews.csv')

# # Load tokenizer and model
# model_name = "nielsr/coref-bert-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# # Use the 'Text' column for text classification and the 'Score' column as the label
# def preprocess_data(examples):
#     tokenized_inputs = {'input_ids': [], 'attention_mask': [], 'labels': []}
#     for text, score in zip(examples['Text'], examples['Score']):
#         tokenized_input = tokenizer(text, truncation=True, max_length=512, padding='max_length')
#         tokenized_inputs['input_ids'].append(tokenized_input['input_ids'])
#         tokenized_inputs['attention_mask'].append(tokenized_input['attention_mask'])
#         tokenized_inputs['labels'].append(int(score) - 1)  # Assuming scores are 1-5, convert to 0-4
#     return tokenized_inputs

# # Preprocess the dataset
# tokenized_datasets = dataset.map(preprocess_data, batched=True)

# # Split the dataset into training and testing sets
# train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(8000))
# test_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(8000, 10000))

# # Define metrics
# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)
#     true_labels = [[label for label in labels[i] if label != -100] for i in range(len(labels))]
#     true_predictions = [
#         [pred for (pred, label) in zip(prediction, labels[i]) if label != -100]
#         for i, prediction in enumerate(predictions)
#     ]
#     results = precision_recall_fscore_support(np.concatenate(true_labels), np.concatenate(true_predictions), average='weighted')
#     accuracy = np.mean(np.concatenate(true_labels) == np.concatenate(true_predictions))
#     return {
#         'precision': results[0],
#         'recall': results[1],
#         'f1': results[2],
#         'accuracy': accuracy,
#     }

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# # Initialize trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     compute_metrics=compute_metrics,
# )

# # Train and evaluate
# trainer.train()
# eval_results = trainer.evaluate()
# print(eval_results)



from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# Load your dataset from a CSV file
dataset = load_dataset('csv', data_files='Reviews.csv')

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Use the 'Text' column for text classification and the 'Score' column as the label
def preprocess_data(examples):
    tokenized_inputs = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for text, score in zip(examples['Text'], examples['Score']):
        tokenized_input = tokenizer(text, truncation=True, max_length=512, padding='max_length')
        tokenized_inputs['input_ids'].append(tokenized_input['input_ids'])
        tokenized_inputs['attention_mask'].append(tokenized_input['attention_mask'])
        tokenized_inputs['labels'].append(int(score) - 1)  # Assuming scores are 1-5, convert to 0-4
    return tokenized_inputs

# Preprocess the dataset
tokenized_datasets = dataset.map(preprocess_data, batched=True)

# Split the dataset into training and testing sets
train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(8000))
test_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(8000, 10000))

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
    accuracy = np.mean(np.concatenate(true_labels) == np.concatenate(true_predictions))
    return {
        'precision': results[0],
        'recall': results[1],
        'f1': results[2],
        'accuracy': accuracy,
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
eval_results = trainer.evaluate()
print(eval_results)

# Test the model with a sentence
test_sentence = "This product is amazing!"
test_inputs = tokenizer(test_sentence, truncation=True, max_length=512, padding='max_length', return_tensors="pt")
test_outputs = model(**test_inputs)
test_prediction = np.argmax(test_outputs.logits.detach().numpy())
print(f"Test sentence: {test_sentence}")
print(f"Predicted label: {test_prediction}")