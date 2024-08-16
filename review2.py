from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


dataset = load_dataset('csv', data_files='Review2.csv')

# testing on this model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)


# update acc/ to req
# Use the 'Text' column for text classification and the 'Score' column as the label
def preprocess_data(examples):
    tokenized_inputs = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for text, score in zip(examples['Text'], examples['Score']):
        tokenized_input = tokenizer(text, truncation=True, max_length=512, padding='max_length')
        tokenized_inputs['input_ids'].append(tokenized_input['input_ids'])
        tokenized_inputs['attention_mask'].append(tokenized_input['attention_mask'])
        # changed score to 0-4 range/value
        tokenized_inputs['labels'].append(int(score) - 1)  
    return tokenized_inputs

# Preprocess the dataset
tokenized_datasets = dataset.map(preprocess_data, batched=True)

# Split the dataset into training and testing sets
# train_dataset = tokenized_datasets.shuffle(seed=42).select(range(200))
# test_dataset = tokenized_datasets.shuffle(seed=42).select(range(200, 260))

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
test_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200, 259))

# go with compute
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)  # Change axis=2 to axis=1
    true_labels = labels
    results = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    accuracy = np.mean(true_labels == predictions)
    return {
        'precision': results[0],
        'recall': results[1],
        'f1': results[2],
        'accuracy': accuracy,
    }

# Define training arguments
# change acc./
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
)

# print(training_args)

# Initialize trainer
trainer = Trainer(
    model=model,
    # print(model)
    args=training_args,
    train_dataset=train_dataset,
    # print(train_dataset)
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
eval_results = trainer.evaluate()
# eval_results
print(eval_results)

# Test the model with a sentence
test_sentence = "This product is amazing!"

# test_sentence ="bad product"
# test_sentence ="These Nature Valley Nut Lovers Variety Pack was perfect."

test_inputs = tokenizer(test_sentence, truncation=True, max_length=512, padding='max_length', return_tensors="pt")
test_outputs = model(**test_inputs)
test_prediction = np.argmax(test_outputs.logits.detach().numpy())

# print #values

print(f"Test sentence: {test_sentence}")
#result
print(f"Predicted label: {test_prediction}")