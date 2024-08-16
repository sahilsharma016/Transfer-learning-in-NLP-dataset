from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

dataset = load_dataset('conll2012_ontonotesv5', 'english_v4')


model_name = "nielsr/coref-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=8)

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

validation_dataset = TensorDataset(
    torch.tensor(tokenized_datasets['validation']['input_ids']),
    torch.tensor(tokenized_datasets['validation']['attention_mask']),
    torch.tensor(tokenized_datasets['validation']['labels'])
)

validation_dataloader = DataLoader(validation_dataset, batch_size=8)


def get_predictions(dataloader, model):
    model.eval()
    predictions = []
    labels = []

    for batch in dataloader:
        input_ids, attention_mask, batch_labels = batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())

    return predictions, labels

predictions, labels = get_predictions(validation_dataloader, model)
# predictions, labels = get_predictions(tokenized_datasets['validation'], model)



def compute_metrics(predictions, labels):
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    # Remove padding (-100) from labels
    valid_indices = labels != -100
    predictions = predictions[valid_indices]
    labels = labels[valid_indices]

    results = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = np.mean(labels == predictions)
    return {
        'precision': results[0],
        'recall': results[1],
        'f1': results[2],
        'accuracy': accuracy,
    }

metrics = compute_metrics(predictions, labels)
print(metrics)
