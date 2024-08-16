from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the dataset
dataset = load_dataset("conll2012_ontonotesv5",'english_v4', split="train[:1%]")

# Load the pre-trained model and tokenizer
model_name = "nielsr/coref-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Function to tokenize and align labels
def tokenize_and_align_labels(examples):
    all_words = []
    all_labels = []
    
    for sentence in examples['sentences']:
        words = int(sentence['words'])
        labels = sentence['named_entities']
        all_words.extend(words)
        all_labels.extend(labels)
    
    tokenized_inputs = tokenizer(all_words, is_split_into_words=True, padding="max_length", truncation=True, max_length=128)
    
    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = -100
    
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(all_labels[word_idx])
        else:
            labels.append(-100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply the tokenization function to the dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Evaluation function
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

# Evaluate the fine-tuned model
evaluate_model(model, tokenized_dataset)
