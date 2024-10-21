import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import hdbscan
from function_files.kg_functions import get_uri_info, update_kg_ip_data
from function_files.clustering_functions import encode_phrases

def main():
    """
    Prepares the data use to create the knowledge graph by:

    - Loading data: Loads relevant datasets and configuration.
    - Process feature data: Prepares feature data, creates new nodes and relationships for unassigned features, and updates existing relationships.
    - Update URIs: Updates URI information for matched features.
    - Save updated data: Saves the updated datasets to CSV and pickle files.
    Key steps:

    - Loads dataframes and configuration.
    - Prepares feature data by filtering and creating new nodes and relationships.
    - Updates URIs for matched features.
    - Saves updated datasets.
    Purpose:

    This script is designed to initialise the knowledge graph by incorporating the new information with the existing NIST based blueprint.
    """

    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    emb_model_name = config['embedding_model']
    pca_dim = config['pca_dim']
    emb_model = SentenceTransformer(emb_model_name)

    feature_data = pd.read_pickle("datasets/feature_classified")
    match_data =   pd.read_pickle("datasets/match_classified")
    init_uri_data = pd.read_pickle("init_data/init_neo4j/uri_data")
    nodes_data = pd.read_csv("init_data/init_neo4j/nodes.csv")
    feat_uri_data = pd.read_csv("init_data/init_neo4j/feat_uri_data.csv")
    feat_cluster_data = pd.read_csv("init_data/init_neo4j/feat_cluster_data.csv")

    resource_info = pd.read_pickle("dbpedia/resource_info")

    feat_data = feature_data[feature_data['feature_label']==1].copy()
    cluster_df = feat_data[['cluster_label', 'Derived Phrase']].copy()
    feat_uri_df = feat_uri_data.copy()
    feat_cluster_df = feat_cluster_data.copy()
    cluster_df.drop_duplicates(subset=['cluster_label', 'Derived Phrase'], inplace=True)
    cluster_df.reset_index(drop=True, inplace=True)
    cluster_df.to_csv('neo4j/cluster_data.csv', index=False)

    uri_data = match_data[match_data['match_label'] == 1].copy()
    uri_data = uri_data[['URIs', 'Names', 'Abstracts', 
        'cluster_label', 'Derived Phrase']]
    uri_data[['Genre(s)','GenreNames','GenreOf','GenreOfNames','dbo_type(s)','Developer(s)','Website(s)','Disambiguate(s)','DisambiguateOf','DisambiguateOfNames','Product(s)','ProductNames','ProductOf','ProductOfNames','LinkedResources','LinkedNames']] = uri_data['URIs'].apply(lambda x: get_uri_info(x, resource_info, 0))
    uri_data.rename(columns={'URIs':'Resource', 'Names':'Label', 'Abstracts':'Abstract'}, inplace=True)
    uri_df = pd.concat([init_uri_data, uri_data], ignore_index=True)
    uri_df.drop_duplicates(subset=['Resource'], keep='first', inplace=True)
    uri_df.reset_index(drop=True, inplace=True)

    init_clusters = feat_cluster_data[feat_cluster_data['Cluster_ID'].apply(lambda x: not pd.isna(x))]['Cluster_ID'].astype(int).unique()
    all_names = feat_data[feat_data['cluster_label'].apply(lambda x: x not in init_clusters)]['Derived Phrase'].unique()
    # Create the nodes dataframe
    nodes_df = pd.DataFrame(all_names, columns=['Name'])
    # Add a unique ID for each node
    nodes_df['Feature_ID'] = 'feature_' + nodes_df['Name'].str.replace(' ', '_').str.replace(r'[^\w\(\)]', '', regex=True)
    
    for idx, row in nodes_df.iterrows():
        name = row['Name']
        id = row['Feature_ID']
        cluster = feat_data.loc[(feat_data['Derived Phrase']==name), 'cluster_label'].values[0]
        feat_cluster_row = pd.DataFrame({'Feature_ID': [id],  'Cluster_ID' : [cluster]})
        feat_cluster_df = pd.concat([feat_cluster_df, feat_cluster_row], ignore_index=True)
        uri_data_rows = uri_data[uri_data['Derived Phrase'] == name]
        for idx, row in uri_data_rows.iterrows():
            uri = row['Resource']
            feat_uri_row = pd.DataFrame({'Feature_ID': [id],  'URI' : [uri]})
            feat_uri_df = pd.concat([feat_uri_df, feat_uri_row], ignore_index=True)
        
    feat_cluster_df.drop_duplicates(subset=['Feature_ID', 'Cluster_ID'], keep='first', inplace=True)
    feat_cluster_df.reset_index(drop=True, inplace=True)
    feat_uri_data.drop_duplicates(subset=['Feature_ID', 'URI'], keep='first', inplace=True)
    feat_uri_data.reset_index(drop=True, inplace=True)
    nodes_df = pd.concat([nodes_data, nodes_df], ignore_index=True)
    nodes_df.drop_duplicates(subset=['Feature_ID'], keep='first', inplace=True)
    nodes_df.reset_index(drop=True, inplace=True)

    nodes_df.to_csv("neo4j/nodes.csv", index=False)
    feat_uri_df.to_csv("neo4j/feat_uri_data.csv", index=False)
    feat_cluster_df.to_csv("neo4j/feat_cluster_data.csv", index=False)
    uri_df.to_csv("neo4j/uri_data.csv", index=False)
    uri_df.to_pickle("neo4j/uri_data")
    update_kg_ip_data(cluster_df)


    feat_data['embeddings'] = list(encode_phrases(feat_data['Derived Phrase'].tolist(), emb_model))
    embeddings = np.stack(feat_data['embeddings'].values)
    pca = PCA(n_components=pca_dim) 
    reduced_embeddings = pca.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, cluster_selection_method='eom')
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    feat_data['cluster_label_2'] = cluster_labels
    feat_data.to_pickle('datasets/features_clustered')

    feature_data_clusters = feat_data.reset_index().groupby('cluster_label_2').agg({
    'Derived Phrase' : list,
    'cluster_label': list
    }).reset_index()

    feature_data_clusters.to_csv("cluster_data/feature_data_clusters.csv", index=False)
    feature_data_clusters.to_pickle("cluster_data/feature_data_clusters")


if __name__ == '__main__':
    main()