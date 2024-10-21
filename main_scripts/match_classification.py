import pandas as pd
import json
from tqdm import tqdm
import os
import pickle
# import google.generativeai as genai
# import vertexai
from vertexai.generative_models import GenerativeModel
from function_files.classifier_functions import load_prompt_template, create_match_prompt, get_gemini_response


def main():
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    gemini_model_name = config['gemini_model_name']
    match_prompt_select = config['match_prompt_select']

    pickle_file = 'datasets/match_data_progress'
    # Check if the pickle file exists and load it, otherwise use original data
    if os.path.exists(pickle_file):
        print("Loading progress from pickle file...")
        with open(pickle_file, 'rb') as f:
            match_data = pickle.load(f)
    else:
        # If no pickle file exists, start with the original data
        match_data = pd.read_pickle('datasets/match_data')
        match_prompt_template = load_prompt_template(f"few-shot_prompts/match_prompts/prompt_{match_prompt_select}.txt")
        match_data['Prompt'] = match_data.apply(lambda row: create_match_prompt(row, match_prompt_template), axis=1)
        match_data['Gemini Response'] = None

    # project_id = ""
    # vertexai.init(project=project_id, location="us-central1")

    model = GenerativeModel(model_name=gemini_model_name)

    try:
    # Iterate over the DataFrame, resuming from where 'Gemini Response' is None
        for index, row in tqdm(match_data.iterrows(), total=match_data.shape[0]):
            if row['Gemini Response'] is None:  # Only process if the response is None
                try:
                    response = get_gemini_response(row, model)
                    match_data.at[index, 'Gemini Response'] = response
                except Exception as e:
                    print(f"Failed at index {index} due to {e}")
            # Save progress periodically (e.g., every 100 rows)
            if index % 50 == 0:
                with open(pickle_file, 'wb') as f:
                    pickle.dump(match_data, f)
        match_data['match_label'] = match_data['Gemini Response'].apply(lambda x: 1 if 'True' in x else 0 if not pd.isna(x) else None)        
        match_data.to_pickle('datasets/match_classified')
        match_data.to_csv('datasets/match_classified.csv', index=False)
    except KeyboardInterrupt:
        print("\nProcess interrupted! Saving progress...")
        with open(pickle_file, 'wb') as f:
            pickle.dump(match_data, f)
        print("Progress saved. Exiting gracefully.")
    except Exception as e:
        print(f"Error occurred: {e}")
        # Save the DataFrame before exiting in case of error
        with open(pickle_file, 'wb') as f:
            pickle.dump(match_data, f)


if __name__ == '__main__':
    main()