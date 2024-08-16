from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


model_name = "nielsr/coref-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# print(model)

from datasets import Dataset

# # Example dataset
data = {
    "text": [
        "John has a dog. He loves his dog.",
        "Mary and her brother went to the park. She enjoyed it.",
    ],
    "labels": [
        [1, 1, 0, 0, 1, 0, 1],  # Toy example labels for coreference
        [1, 1, 1, 0, 0, 0, 1, 0, 1],
    ]
}

dataset = Dataset.from_dict(data)
print(dataset)

def tokenize_example(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(tokenize_example, batched=True)


# import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    labels = []
    for i, label in enumerate(examples["labels"]):
        padded_label = label + [0] * (128 - len(label))
        labels.append(padded_label)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)




# # 
# from transformers import Trainer, TrainingArguments

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     warmup_steps=10,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
#     evaluation_strategy="epoch"
# )

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     eval_dataset=tokenized_dataset,
# )

# # Fine-tune the model
# trainer.train()

# # 

def evaluate_model(model, tokenized_dataset):
    model.eval()
    all_predictions = []
    all_labels = []

    for i in range(len(tokenized_dataset)):
        inputs = {key: torch.tensor([val]) for key, val in tokenized_dataset[i].items() if key in tokenizer.model_input_names}
        labels = torch.tensor([tokenized_dataset[i]['labels']])

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).flatten().tolist()

        all_predictions.extend(predictions)
        all_labels.extend(labels.flatten().tolist())

    precision = precision_score(all_labels, all_predictions, average='micro')
    recall = recall_score(all_labels, all_predictions, average='micro')
    f1 = f1_score(all_labels, all_predictions, average='micro')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Evaluate the model
evaluate_model(model, tokenized_dataset)
