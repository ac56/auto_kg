import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import hdbscan
from function_files.clustering_functions import encode_phrases, save_clusterer, compute_ngrams, derive_most_representative_ngram




def main():
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    emb_model_name = config['embedding_model']
    pca_dim = config['pca_dim']
    min_cluster_size = config['min_cluster_size']
    min_samples = config['min_samples']
    n_min = config['n_min']
    n_max = config['n_max']
    min_freq = config['min_freq']

    phrase_data = pd.read_pickle('web_data/phrase_data')
    emb_model = SentenceTransformer(emb_model_name)

    # Apply the function and convert each embedding vector to a list (if necessary)
    phrase_data['embeddings'] = list(encode_phrases(phrase_data['Processed Phrase'].tolist(), emb_model))

    embeddings = np.stack(phrase_data['embeddings'].values)
    pca = PCA(n_components=pca_dim)  # Adjust `n_components` based on your needs
    reduced_embeddings = pca.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_method='eom')
    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    # Assume `clusterer` is your HDBSCAN instance after fitting
    save_clusterer(clusterer)

    phrase_data['cluster_label'] = cluster_labels

    phrase_data.to_pickle('cluster_data/phrase_data')
    phrase_data.to_csv('cluster_data/phrase_data.csv')

    # phrase_data = pd.read_pickle('cluster_data/phrase_data')

    cluster_data = phrase_data.reset_index().groupby('cluster_label').agg({
        'index' : list,
        'Alternative': list,
        'Header' : list,
        'Context' : list,
        'Original Phrase': list,
        'Lower Phrase' : list,
        'Processed Phrase': list
    }).reset_index()

    # Rename columns for clarity
    cluster_data.columns = ['cluster_label', 'Indices', 'Alternatives', 'Headers', 'Contexts',  'Phrases', 'Lower Phrases', 'Processed Phrases']
    # Exclude unclustered phrases
    cluster_data = cluster_data[(cluster_data['cluster_label'] != -1)].reset_index(drop=True) 

    cluster_data = cluster_data.copy()[cluster_data['Alternatives'].apply(lambda x: len(set(x)) > 1)]
    cluster_data.reset_index(drop=True, inplace=True)

    cluster_data['N-grams'] = cluster_data.apply(lambda row: compute_ngrams(row, n_min, n_max, min_freq=min_freq), axis=1)

    cluster_data['Derived Phrase'] = cluster_data.apply(derive_most_representative_ngram, axis=1)
    cluster_data['Embedding'] = list(encode_phrases(cluster_data['Derived Phrase'].tolist(), emb_model))

    cluster_data.to_pickle('cluster_data/cluster_data')
    cluster_data.to_csv('cluster_data/cluster_data.csv')

if __name__ == '__main__':
    main()