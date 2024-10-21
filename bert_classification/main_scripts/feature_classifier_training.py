import os
import pandas as pd
import json
import torch

from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from function_files.bert_functions import FeatureDataset, compute_metrics


def main():
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    # Check if CUDA is available and use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config['bert_feature_model_name']
      

    feature_data = pd.read_pickle('datasets/feature_data')

    # Prepare the dataset by combining 'Derived Phrase', 'Sel Contexts', and 'cluster_label'
    X = feature_data[['Derived Phrase', 'Sel Contexts', 'cluster_label']]
    y = feature_data['feature_label'].astype(int)

    # Split data into training and validation sets, including 'cluster_label'
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Prepare the text inputs   for the tokenizer
    X_train_text = X_train['Derived Phrase'] + " [SEP] " + X_train['Sel Contexts'].apply(lambda x: ' '.join(x))
    X_val_text = X_val['Derived Phrase'] + " [SEP] " + X_val['Sel Contexts'].apply(lambda x: ' '.join(x))

    # Create datasets
    train_dataset = FeatureDataset(X_train_text.values, y_train.values, tokenizer, max_len=256)
    val_dataset = FeatureDataset(X_val_text.values, y_val.values, tokenizer, max_len=256)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='bert_classification/feature_classifier/results',  # Directory where the model checkpoints will be saved
        num_train_epochs=3,  # Total number of training epochs
        per_device_train_batch_size=16,  # Batch size for training
        per_device_eval_batch_size=16,  # Batch size for evaluation
        warmup_steps=500,  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # Strength of weight decay
        logging_dir='bert_classification/feature_classifier/logs',  # Directory for storing logs
        logging_steps=10,  # Log every 10 steps
        eval_strategy="steps",  # Evaluate every `eval_steps`
        save_steps=500,  # Save checkpoint every 500 steps
        eval_steps=500,  # Evaluate every 500 steps
        save_total_limit=2,  # Only keep the last 2 checkpoints
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Train model
    trainer.train()

    # Evaluate model
    results = trainer.evaluate()
    print(f"Validation Accuracy: {results['eval_accuracy']:.2f}")

    # Save the final model
    model.save_pretrained('bert_classification/feature_classifier/final_model')
    tokenizer.save_pretrained('bert_classification/feature_classifier/final_model_tokenizer')

if __name__ == '__main__':
    main()