import pandas as pd
import pysbd
import json
from function_files.dataset_functions import remove_non_english_characters, get_sentences, choose_contexts


def main():
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    n_abst_sent = config['n_abst_sent']
    n_contexts = config['n_contexts']

    init_dataset = pd.read_pickle('datasets/init_dataset')
    seg = pysbd.Segmenter(language="en", clean=False)


    match_dataset = init_dataset[init_dataset['Abstracts'].apply(lambda x: x not in ["No abstract available","No abstract found"])]
    match_dataset.reset_index(drop=True, inplace=True)
    match_dataset['Abstract Sent'] = match_dataset['Abstracts'].apply(lambda x: remove_non_english_characters(" ".join(get_sentences(x, seg)[:n_abst_sent])))

    feature_dataset = init_dataset.drop_duplicates(subset=['cluster_label'], keep='first').copy()
    feature_dataset.reset_index(drop=True, inplace=True)

    feature_dataset['Sel Contexts'] = feature_dataset.apply(lambda row: choose_contexts(row, n_contexts), axis=1) 
    context_map = dict(zip(feature_dataset['cluster_label'], feature_dataset['Sel Contexts']))
    match_dataset['Sel Contexts'] = match_dataset['cluster_label'].apply(lambda label: context_map.get(label, match_dataset.loc[match_dataset['cluster_label'] == label, 'Contexts'].values[0]))
   
    feature_dataset.to_pickle('datasets/feature_data')
    match_dataset.to_pickle('datasets/match_data')

    feature_dataset.to_csv('datasets/feature_data.csv', index=False)
    match_dataset.to_csv('datasets/match_data.csv', index=False)


if __name__ == '__main__':
    main()