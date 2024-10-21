import pandas as pd
import json
from tqdm import tqdm
import os
import pickle
# import google.generativeai as genai
# import vertexai
from vertexai.generative_models import GenerativeModel
from function_files.classifier_functions import load_prompt_template, create_feature_prompt, get_gemini_response


def main():
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    gemini_model_name = config['gemini_model_name']
    feature_prompt_select = config['feature_prompt_select']

    pickle_file = 'datasets/feature_data_progress'
    # Check if the pickle file exists and load it, otherwise use original data
    if os.path.exists(pickle_file):
        print("Loading progress from pickle file...")
        with open(pickle_file, 'rb') as f:
            feature_data = pickle.load(f)
    else:
        # If no pickle file exists, start with the original data
        feature_data = pd.read_pickle('datasets/feature_data')
        feature_prompt_template = load_prompt_template(f"few-shot_prompts/feature_prompts/prompt_{feature_prompt_select}.txt")
        feature_data['Prompt'] = feature_data.apply(lambda row: create_feature_prompt(row, feature_prompt_template), axis=1)
        feature_data['Gemini Response'] = None


    # project_id = ""
    # vertexai.init(project=project_id, location="us-central1")

    model = GenerativeModel(model_name=gemini_model_name)

    try:
    # Iterate over the DataFrame, resuming from where 'Gemini Response' is None
        for index, row in tqdm(feature_data.iterrows(), total=feature_data.shape[0]):
            if row['Gemini Response'] is None:  # Only process if the response is None
                try:
                    response = get_gemini_response(row, model)
                    feature_data.at[index, 'Gemini Response'] = response
                except Exception as e:
                    print(f"Failed at index {index} due to {e}")
            # Save progress periodically (e.g., every 100 rows)
            if index % 50 == 0:
                with open(pickle_file, 'wb') as f:
                    pickle.dump(feature_data, f)
        
        feature_data['feature_label'] = feature_data['Gemini Response'].apply(lambda x: 1 if 'True' in x else 0 if not pd.isna(x) else None)        
        feature_data.to_pickle('datasets/feature_classified')
        feature_data.to_csv('datasets/feature_classified.csv', index=False)
    except KeyboardInterrupt:
        print("\nProcess interrupted! Saving progress...")
        with open(pickle_file, 'wb') as f:
            pickle.dump(feature_data, f)
        print("Progress saved. Exiting gracefully.")
    except Exception as e:
        print(f"Error occurred: {e}")
        # Save the DataFrame before exiting in case of error
        with open(pickle_file, 'wb') as f:
            pickle.dump(feature_data, f)


if __name__ == '__main__':
    main()