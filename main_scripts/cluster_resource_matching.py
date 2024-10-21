import pandas as pd
import numpy as np
import json
import itertools
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from function_files.dbpedia_functions import encode_phrases, predict_cluster_dbpedia, update_resource_info, get_new_dbpedia_vocab, get_abstract, update_abstracts, find_most_similar, calc_name_cos_sim, dbpedia_row_search
from function_files.clustering_functions import load_clusterer

def main():
    """
    Processing DBpedia data and matching it to relevant clusters from a pre-existing dataset. It leverages techniques like sentence embeddings, PCA, and cosine similarity to achieve this.

    Key steps:

    - Load data and configurations: Loads necessary dataframes and configuration settings.
    - Process DBpedia vocabulary: Encodes DBpedia names into embeddings, assigns cluster labels, and saves the updated vocabulary.
    - Match DBpedia entries to clusters: Calculates cosine similarity between DBpedia embeddings and cluster embeddings, assigns matching clusters, and stores relevant information.
    - Build output DataFrame: Constructs a DataFrame containing matched DBpedia entries, their cluster labels, and associated information.
    - Update abstracts: Fetches missing abstracts from an external resource.
    - Save output: Saves the final processed DataFrame to pickle and CSV formats.
    Purpose:

    The code aims to enrich DBpedia data by associating it with relevant clusters from a pre-existing dataset.
    """

    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    emb_model_name = config['embedding_model']
    pca_dim = config['pca_dim']
    sim_threshold = config['sim_threshold']

    clusterer = load_clusterer()
    phrase_data = pd.read_pickle('cluster_data/phrase_data')
    cluster_data = pd.read_pickle('cluster_data/cluster_data')
    resource_info = pd.read_pickle('dbpedia/resource_info')

    embeddings = np.stack(phrase_data['embeddings'].values)
    emb_model = SentenceTransformer(emb_model_name)
    pca = PCA(n_components=pca_dim)  # Adjust `n_components` based on your needs
    reduced_embeddings = pca.fit_transform(embeddings)

    dbpedia_vocab = pd.read_pickle('dbpedia/dbpedia_vocab')
    dbpedia_vocab['embeddings'] = list(encode_phrases(dbpedia_vocab['Name'].tolist(), emb_model))
    dbpedia_vocab['cluster_label'] = dbpedia_vocab.apply(lambda row: predict_cluster_dbpedia(row, clusterer, pca, reduced_embeddings), axis=1)
    clustered_resources = dbpedia_vocab[dbpedia_vocab['cluster_label'] != -1]
    new_uris = [uri for uri in list(clustered_resources['URI']) if uri not in list(resource_info['Resource'].to_list())]
    resource_info = update_resource_info(new_uris, resource_info)

    new_dbpedia_vocab = get_new_dbpedia_vocab(resource_info, dbpedia_vocab)  
    # Apply the function and convert each embedding vector to a list (if necessary)
    new_dbpedia_vocab['embeddings'] = list(encode_phrases(new_dbpedia_vocab['Name'].tolist(), emb_model))
    new_dbpedia_vocab['cluster_label'] = new_dbpedia_vocab.apply(lambda row: predict_cluster_dbpedia(row, clusterer, pca, reduced_embeddings), axis=1)
    dbpedia_vocab = pd.concat([dbpedia_vocab, new_dbpedia_vocab], ignore_index=True)

    print('Saving new DBPedia vocabulary...')
    dbpedia_vocab.to_csv('dbpedia/dbpedia_vocab.csv', index=False)
    dbpedia_vocab.to_pickle('dbpedia/dbpedia_vocab')

    # Extracting embeddings into a numpy array for faster operation
    cluster_embeddings = np.stack(cluster_data['Embedding'])
    cluster_embeddings = pca.transform(cluster_embeddings)

    
    # Initialize new columns
    dbpedia_vocab['Abstract'] = dbpedia_vocab['URI'].apply(lambda x: get_abstract(x, resource_info))
    match_1 = dbpedia_vocab.copy()
    match_1['match_cluster_label'] = None
    match_1['Derived Phrase'] = None
    match_1['Cos Sim'] = None

    # Iterate over match_1 and calculate similarities
    for index, row in match_1.iterrows():
        most_similar_index, max_similarity = find_most_similar(pca.transform(row['embeddings'].reshape(1, -1)), cluster_embeddings)
        # Assign the most similar details from feature_doc_df
        match_1.at[index, 'match_cluster_label'] = cluster_data.iloc[most_similar_index]['cluster_label']
        match_1.at[index, 'Derived Phrase'] = cluster_data.iloc[most_similar_index]['Derived Phrase']
        match_1.at[index, 'Cos Sim'] = max_similarity
    match_1 = match_1.loc[(match_1['cluster_label']!=-1) & (match_1['Cos Sim'] > sim_threshold)].reset_index(drop=True)

    # Initialize a dictionary to hold the data
    data_dict = {label: {'URIs': [], 'Names': [], 'Abstracts': []}
            for label in cluster_data['cluster_label'].unique()}
    
    valid_clusters = list(data_dict.keys())
    match_1 = match_1[match_1['match_cluster_label'].isin(valid_clusters)]
    
    for index, row in match_1.iterrows():
            data_dict[row['match_cluster_label']]['URIs'].append(row['URI'])
            data_dict[row['match_cluster_label']]['Names'].append(row['Name']if pd.notna(row['Name']) else 'No Name')
            data_dict[row['match_cluster_label']]['Abstracts'].append(row['Abstract'] if pd.notna(row['Abstract']) else 'No Abstract')
    
    dbpedia_df = cluster_data.merge(dbpedia_vocab[['cluster_label','URI', 'Name', 'Abstract']], 
              on='cluster_label', 
              how='left')
    dbpedia_df['Name Avg Cos Sim'] = dbpedia_df.apply(lambda row: calc_name_cos_sim(row, 'Name', phrase_data, dbpedia_vocab, emb_model, pca), axis=1)
    # Process each row and populate the dictionary
    for index, row in dbpedia_df.iterrows():
        if row['Name Avg Cos Sim'] > sim_threshold:
            data_dict[row['cluster_label']]['URIs'].append(row['URI'])
            data_dict[row['cluster_label']]['Names'].append(row['Name']if pd.notna(row['Name']) else 'No Name')
            data_dict[row['cluster_label']]['Abstracts'].append(row['Abstract'] if pd.notna(row['Abstract']) else 'No Abstract')
            


    cluster_dbp_search = cluster_data.copy()[['cluster_label', 'Indices', 'Phrases', 'Derived Phrase']]
    cluster_dbp_search[['URI', 'Name', 'Name Avg Cos Sim']] = cluster_dbp_search.apply(lambda row: dbpedia_row_search(row, phrase_data, emb_model, pca), axis=1)
    cluster_dbp_search['Abstract'] = 'TBD'

    # Process each row and populate the dictionary
    for index, row in cluster_dbp_search.iterrows():
        if row['Name Avg Cos Sim'] > sim_threshold:
            data_dict[row['cluster_label']]['URIs'].append(row['URI'])
            data_dict[row['cluster_label']]['Names'].append(row['Name']if pd.notna(row['Name']) else 'No Name')
            data_dict[row['cluster_label']]['Abstracts'].append(row['Abstract'] if pd.notna(row['Abstract']) else 'No Abstract')

    
    df_list = []
    # Loop through each cluster_label in the dictionary
    for cluster_label, contents in data_dict.items():
        # Create a DataFrame from each part of the dictionary for this cluster_label
        if len(set(map(len, [contents['URIs'], contents['Names'], contents['Abstracts']]))) == 1:
            temp_df = pd.DataFrame({
                'URIs': contents['URIs'],
                'Names': contents['Names'],
                'Abstracts': contents['Abstracts']
            })
            temp_df['cluster_label'] = cluster_label  # Add the cluster_label as a column
            df_list.append(temp_df)
        else:
            print(f"Mismatch in data lengths for cluster_label {cluster_label}")

    # Concatenate all DataFrames
    dbpedia_data_df = pd.concat(df_list, ignore_index=True)
    dbpedia_data_df = dbpedia_data_df.merge(cluster_data[['cluster_label', 'Alternatives', 'Headers', 'Contexts', 'Phrases', 'Derived Phrase']], 
              on='cluster_label', 
              how='left')
    dbpedia_data_df.drop_duplicates(subset=['URIs', 'cluster_label'], inplace=True)

    miss_info = dbpedia_data_df[dbpedia_data_df['Abstracts']== 'TBD']
    new_uris = [uri for uri in list(miss_info['URIs']) if uri not in list(resource_info['Resource'].to_list())]
    resource_info = update_resource_info(new_uris, resource_info)

    
    redirects = set(list(itertools.chain.from_iterable(resource_info['Redirects'].dropna())))
    new_uris = [re for re in redirects if re not in resource_info['Resource'].unique()]
    resource_info = update_resource_info(new_uris, resource_info)

    # dbpedia_data_df.to_pickle('dbpedia/temp_dbpedia_data_df')
    # dbpedia_data_df = pd.read_pickle('dbpedia/temp_dbpedia_data_df')

    dbpedia_data_df['Abstracts'] = dbpedia_data_df.apply(lambda row: update_abstracts(row, resource_info), axis=1)

    dbpedia_data_df.to_pickle('datasets/init_dataset')
    dbpedia_data_df.to_csv('datasets/init_dataset.csv',index=False)


if __name__ == '__main__':
    main()