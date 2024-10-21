import pandas as pd
import torch
from torch.utils.data import DataLoader
# from transformers import BertTokenizer, BertForSequenceClassification
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from function_files.bert_functions import MatchDataset


def main():
    # Check if CUDA is available and use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match_data = pd.read_pickle('datasets/match_data')

    # Combine 'Abstract Sent' and 'Names' for DBPedia context
    match_data['dbpedia_context'] = match_data['Names'] + ". " + match_data['Abstract Sent']

    # Combine 'Sel Contexts' and 'Derived Phrase' for web text context
    match_data['web_context'] = match_data['Derived Phrase'] + ". " + match_data['Sel Contexts'].apply(lambda x: ' '.join(x))

    # Path to the saved model and tokenizer
    model_path = 'bert_classification/match_classifier/final_model'
    tokenizer_path = f'bert_classification/match_classifier/final_model_tokenizer'

    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # model = AutoModelForSequenceClassification.from_pretrained(model_path)  

    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)

    # Prepare the text inputs   for the tokenizer
    X_text = match_data['dbpedia_context'] + " [SEP] " + match_data['web_context']

    # Create dataset for inference
    inference_dataset = MatchDataset(X_text.values, None, tokenizer, max_len=256)  # No labels needed for inference
    inference_loader = DataLoader(inference_dataset, batch_size=32, shuffle=False)

    # Set the model to evaluation mode
    model.eval()

    # Store predictions
    predictions = []

    # Inference loop
    with torch.no_grad():
        for batch in inference_loader:
            # Get the input IDs and attention masks from the batch
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get the logits (unnormalized scores) from the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Get the predicted labels by taking the argmax of the logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)

    # Add the predictions as a new column to match_data
    match_data['match_label'] = predictions

    # Save the updated dataframe (optional)
    match_data.to_pickle('bert_classification/results/match_classified')

if __name__ == '__main__':
    main()