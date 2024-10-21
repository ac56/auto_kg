import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from function_files.kg_functions import get_uri_info, get_node_depth, update_kg_ip_data

def main():
    """
    updates a knowledge graph by:

    - Loading data: Loads relevant datasets and configuration.
    - Identifying new clusters and features: Identifies clusters and features not yet present in the knowledge graph.
    - Assigning new clusters to existing features: Calculates similarities between new cluster embeddings and existing feature embeddings. Assigns new clusters to existing features based on similarity thresholds.
    - Create new nodes and relationships: For unassigned features, creates new nodes and relationships in the knowledge graph.
    - Update URIs and abstracts: Updates URI information and abstracts for matched features.
    - Save updated data: Saves the updated datasets to CSV and pickle files.
    Key steps:

    - Loads dataframes and configuration.
    - Identifies new clusters and features.
    - Calculates cosine similarities between embeddings.
    - Assigns new clusters to existing features based on similarity thresholds.
    - Creates new nodes and relationships for unassigned features.
    - Updates URIs and abstracts.
    - Saves updated datasets.
    Purpose:

    This script is designed to expand and refine a knowledge graph by incorporating new features and clusters, ensuring the graph remains up-to-date and accurate.
    """

    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    feature_data = pd.read_pickle("datasets/feature_classified")
    match_data =   pd.read_pickle("datasets/match_classified")
    matched_data = match_data[match_data['match_label']==1].copy()
    features_clustered = pd.read_pickle('datasets/features_clustered')
    resource_info = pd.read_pickle('dbpedia/resource_info')
    feat_df = feature_data[feature_data['feature_label']==1].copy()
    feature_data_clusters = pd.read_pickle("cluster_data/feature_data_clusters")
    nodes_df = pd.read_csv('neo4j/nodes.csv')
    feat_cluster_df = pd.read_csv('neo4j/feat_cluster_data.csv', dtype={'Cluster_ID': 'Int64'})
    feat_uri_df = pd.read_csv('neo4j/feat_uri_data.csv')
    hierarchy_df = pd.read_csv('init_data/init_neo4j/hierarchy.csv')
    cluster_df = pd.read_csv('neo4j/cluster_data.csv')
    uri_df = pd.read_pickle('neo4j/uri_data')
    

    current_clusters = feat_cluster_df['Cluster_ID'].unique()
    current_clusters = current_clusters[~np.isnan(current_clusters)].astype(np.int64)
    current_nodes_ids = feat_cluster_df['Feature_ID'].unique()
    current_nodes = nodes_df[nodes_df['Feature_ID'].apply(lambda x: x in current_nodes_ids)]['Name'].values
    new_feat_df = feat_df[feat_df['cluster_label'].apply(lambda x: x not in current_clusters)].copy()
    new_clusters = new_feat_df['cluster_label'].unique()

    clusters_to_add = [cl for cl in current_clusters if cl not in cluster_df['cluster_label'].unique()]
    for cluster in clusters_to_add:
        derived_phrase = feature_data[feature_data['cluster_label']==cluster]['Derived Phrase'].values[0]
        cluster_row = pd.DataFrame({'cluster_label':[cluster], 'Derived Phrase':[derived_phrase]})
        cluster_df = pd.concat([cluster_df, cluster_row], ignore_index=True)
        cluster_df.drop_duplicates(subset=['cluster_label', 'Derived Phrase'], inplace=True)
        cluster_df.reset_index(drop=True, inplace=True)

    to_curr_nodes = new_feat_df[new_feat_df['Derived Phrase'].apply(lambda x: x in current_nodes)]
    for index, row in to_curr_nodes.drop_duplicates(subset=['cluster_label', 'Derived Phrase']).iterrows():
        name = row['Derived Phrase']
        cluster = row['cluster_label']

        feature_id = nodes_df.loc[(nodes_df['Name'] == name), 'Feature_ID'].values[0]

        cluster_row = pd.DataFrame({'cluster_label':[cluster], 'Derived Phrase':[name]})
        cluster_df = pd.concat([cluster_df, cluster_row], ignore_index=True)
        cluster_df.drop_duplicates(subset=['cluster_label', 'Derived Phrase'], inplace=True)
        cluster_df.reset_index(drop=True, inplace=True)

        feat_cluster_row = pd.DataFrame({'Feature_ID': [feature_id],  'Cluster_ID' : [cluster]})
        feat_cluster_df = pd.concat([feat_cluster_df, feat_cluster_row], ignore_index=True)
        feat_cluster_df.drop_duplicates(subset=['Feature_ID', 'Cluster_ID'], inplace=True)
        feat_cluster_df.reset_index(drop=True, inplace=True)


    to_curr_nodes_uris = matched_data[matched_data['Derived Phrase'].apply(lambda x: x in current_nodes)]
    for index, row in to_curr_nodes_uris.drop_duplicates(subset=['cluster_label', 'URIs']).iterrows():
        name = row['Derived Phrase']
        cluster = row['cluster_label']
        uri = row['URIs']
        info_rows = resource_info[resource_info['Resource']==uri]    
        if info_rows['Redirects'].notna().any() and len(info_rows['Redirects'].values[0]) > 0:
            uri = info_rows['Redirects'].values[0][0] 
            info_rows = resource_info[resource_info['Resource'] == uri]
        feature_id = nodes_df.loc[(nodes_df['Name'] == name), 'Feature_ID'].values[0]
        current_uris = feat_uri_df[feat_uri_df['Feature_ID']==feature_id]['URI'].values
        if uri not in current_uris:
            feat_uri_row = pd.DataFrame({'Feature_ID': [feature_id],  'URI' : [uri]})
            feat_uri_df = pd.concat([feat_uri_df, feat_uri_row], ignore_index=True)
            feat_uri_df.drop_duplicates(subset=['Feature_ID', 'URI'], inplace=True)
            feat_uri_df.reset_index(drop=True, inplace=True)
        if uri not in uri_df['Resource'].unique():
            columns = ['Resource', 'Label', 'Abstract', 'Genre(s)', 'GenreNames', 'GenreOf', 'GenreOfNames', 'dbo_type(s)', 'Developer(s)', 'Website(s)', 'Disambiguate(s)', 'DisambiguateOf','DisambiguateOfNames', 'Product(s)', 'ProductNames', 'ProductOf', 'ProductOfNames', 'LinkedResources', 'LinkedNames']
            uri_row = info_rows[columns].iloc[0].to_dict()
            # uri_row['cluster_label'] = cluster
            # uri_row['Derived Phrase'] = name
            uri_row = pd.Series(uri_row)
            uri_row = uri_row.reindex(uri_df.columns)
            uri_df.loc[len(uri_df)] = uri_row.values
            uri_df.drop_duplicates(subset=['Resource'], inplace=True)
            uri_df.reset_index(drop=True, inplace=True)

    for index, row in feature_data_clusters[feature_data_clusters['cluster_label_2']!=(-1)].iterrows():
        memb_clusters = row['cluster_label']
        exist = [cl for cl in memb_clusters if cl in current_clusters]
        new = [cl for cl in memb_clusters if cl not in current_clusters]
        if len(new) == 0:
            continue
        new_feats = feat_df.loc[(feat_df['cluster_label'].apply(lambda x: x in new))]
        if len(exist)>0:
            feat_id = feat_cluster_df.loc[(feat_cluster_df['Cluster_ID']==exist[0]), 'Feature_ID'].values[0]
        else:
            node_name = new_feats['Derived Phrase'].values[0]
            feat_id = 'feature_' + node_name.str.replace(' ', '_').str.replace(r'[^\w\(\)]', '', regex=True)
            nodes_row = pd.DataFrame({'Name':[node_name], 'Feature_ID':[feat_id]})
            nodes_df = pd.concat([nodes_df, nodes_row], ignore_index=True)
            nodes_df.drop_duplicates(subset='Name', inplace=True)
            nodes_df.reset_index(drop=True, inplace=True)
            
        for index, row in new_feats.iterrows():
            cluster = row['cluster_label']
            derived_phrase = row['Derived Phrase']

            if row['match_label']== 1:
                uri = row['URIs']
                if uri in resource_info['Resource'].unique():
                    uri_info = get_uri_info(uri, resource_info, 1)
                    new_uri = uri_info['Resource']
                    resource_info = resource_info[resource_info['Resource']==new_uri]
                    columns = ['Resource', 'Label', 'Abstract', 'Genre(s)', 'GenreNames', 'GenreOf', 'GenreOfNames', 'dbo_type(s)', 'Developer(s)', 'Website(s)', 'Disambiguate(s)', 'DisambiguateOf','DisambiguateOfNames', 'Product(s)', 'ProductNames', 'ProductOf', 'ProductOfNames', 'LinkedResources', 'LinkedNames']
                    uri_row = resource_info[columns].iloc[0].to_dict()
                    uri_row['cluster_label'] = cluster
                    uri_row['Derived Phrase'] = derived_phrase
                    uri_row = pd.Series(uri_row)
                    uri_row = uri_row.reindex(uri_df.columns)
                    uri_df.loc[len(uri_df)] = uri_row.values
                    uri_df.drop_duplicates(subset=['Resource'], inplace=True)
                    uri_df.reset_index(drop=True, inplace=True)
                    feat_uri_row = pd.DataFrame({'Feature_ID': [feat_id],  'URI' : [new_uri]})
                    feat_uri_df = pd.concat([feat_uri_df, feat_uri_row], ignore_index=True)
                    feat_uri_df.drop_duplicates(subset=['Feature_ID', 'URI'], inplace=True)
                    feat_uri_df.reset_index(drop=True, inplace=True)

            cluster_row = pd.DataFrame({'cluster_label':[cluster], 'Derived Phrase':[derived_phrase]})
            cluster_df = pd.concat([cluster_df, cluster_row], ignore_index=True)
            cluster_df.drop_duplicates(subset=['cluster_label', 'Derived Phrase'], inplace=True)
            cluster_df.reset_index(drop=True, inplace=True)
            feat_cluster_row = pd.DataFrame({'Feature_ID': [feat_id],  'Cluster_ID' : [cluster]})
            feat_cluster_df = pd.concat([feat_cluster_df, feat_cluster_row], ignore_index=True)
            feat_cluster_df.drop_duplicates(subset=['Feature_ID', 'Cluster_ID'], inplace=True)
            feat_cluster_df.reset_index(drop=True, inplace=True)

    threshold = config['sim_threshold']
    current_clusters = feat_cluster_df['Cluster_ID'].unique()
    current_clusters = current_clusters[~np.isnan(current_clusters)].astype(np.int64)

    new_clusters = [cl for cl in features_clustered['cluster_label'].unique() if cl not in current_clusters]

    exist_feat_df = features_clustered[features_clustered['cluster_label'].apply(lambda x: x in current_clusters)].copy()
    new_feat_df = features_clustered[features_clustered['cluster_label'].apply(lambda x: x in new_clusters)].copy()

    exist_feat_df.reset_index(drop=True, inplace=True)
    new_feat_df.reset_index(drop=True, inplace=True)


    print('\nFinalizing Nodes...')
    while not new_feat_df.empty:
        # print(f'Existing clusters: \t{len(exist_feat_df)}')
        # print(f'New clusters: \t{len(new_feat_df)}')
        exist_embeddings = np.stack(exist_feat_df['embeddings'].values)
        new_embeddings = np.stack(new_feat_df['embeddings'].values)
        similarity_matrix = cosine_similarity(new_embeddings, exist_embeddings)

        # Find the maximum similarity score for each new node to any existing node
        max_similarities = np.max(similarity_matrix, axis=1)
        # Ensure we do not attempt to slice more elements than exist
        num_to_process = min(10, len(max_similarities))
        # Safely select the top N highest similarities where N is num_to_process
        top_10_indices = np.argsort(max_similarities)[-num_to_process:][::-1]


        # Process each new embedding
        for idx in top_10_indices:
            # Dictionary to accumulate scores for nodes that meet the threshold
            node_scores = {}
            # Indices of the top 10 most similar existing nodes for this new node
            top_indices = np.argsort(similarity_matrix[idx])[-10:]
            new_cluster = new_feat_df.loc[idx, 'cluster_label']
            new_name = new_feat_df.loc[idx, 'Derived Phrase']
            # print(f'Processing the top matches for cluster {new_cluster} with name {new_name}')

            # Initialize to track the best match below the threshold
            best_match_below_threshold = {'score': -1, 'node': None, 'cluster': None}

            for i in top_indices:
                similarity = similarity_matrix[idx][i]
                exist_cluster = exist_feat_df.loc[i, 'cluster_label']
                # print(cluster)
                node = feat_cluster_df[feat_cluster_df['Cluster_ID'] == exist_cluster]['Feature_ID'].values[0]
                if similarity > threshold:
                    # Accumulate scores for nodes above the threshold
                    if node in node_scores:
                        node_scores[node] += similarity
                    else:
                        node_scores[node] = similarity
                else:
                    # Update the best fallback node if the current similarity is the highest below the threshold
                    if similarity > best_match_below_threshold['score']:
                        best_match_below_threshold['score'] = similarity
                        best_match_below_threshold['cluster'] = exist_cluster
                        best_match_below_threshold['node'] = node


            # Decide the best node to attach the new cluster to
            if node_scores:
                # If there are nodes with accumulated scores above the threshold
                best_node = max(node_scores, key=node_scores.get)
                # print(f'Best node for this cluster is {best_node} with a total similarity score of {node_scores[best_node]}')
                # print('Adding cluster to existing node...')
                feat_id = best_node

            else:
                # If no nodes meet the threshold, use the best match below threshold
                # print(f'No suitable nodes above the threshold. Best fallback node is based on cluster {best_match_below_threshold["cluster"]} with similarity {best_match_below_threshold["score"]} and node {best_match_below_threshold['node']}')
                # print('Creating new node...')
                feat_id = 'feature_' + new_name.replace(' ', '_')
                nodes_row = pd.DataFrame({'Name':[new_name], 'Feature_ID':[feat_id]})
                nodes_df = pd.concat([nodes_df, nodes_row], ignore_index=True)
                nodes_df.drop_duplicates(subset=['Name'], inplace=True)
                nodes_df.reset_index(drop=True, inplace=True)

            cluster_row = pd.DataFrame({'cluster_label':[new_cluster], 'Derived Phrase':[new_name]})
            cluster_df = pd.concat([cluster_df, cluster_row], ignore_index=True)
            cluster_df.drop_duplicates(subset=['cluster_label', 'Derived Phrase'], inplace=True)
            cluster_df.reset_index(drop=True, inplace=True)
            feat_cluster_row = pd.DataFrame({'Feature_ID': [feat_id],  'Cluster_ID' : [new_cluster]})
            feat_cluster_df = pd.concat([feat_cluster_df, feat_cluster_row], ignore_index=True)
            feat_cluster_df.drop_duplicates(subset=['Feature_ID', 'Cluster_ID'], inplace=True)
            feat_cluster_df.reset_index(drop=True, inplace=True)
            
            match_label = new_feat_df.loc[idx, 'match_label']
            if match_label == 1:
                uri = new_feat_df.loc[idx, 'URIs']
                if uri in resource_info['Resource'].unique():
                    uri_info = get_uri_info(uri, resource_info, 1)
                    new_uri = uri_info['Resource']
                    resource_info = resource_info[resource_info['Resource']==new_uri]
                    columns = ['Resource', 'Label', 'Abstract', 'Genre(s)', 'GenreNames', 'GenreOf', 'GenreOfNames', 'dbo_type(s)', 'Developer(s)', 'Website(s)', 'Disambiguate(s)', 'DisambiguateOf','DisambiguateOfNames', 'Product(s)', 'ProductNames', 'ProductOf', 'ProductOfNames', 'LinkedResources', 'LinkedNames']
                    uri_row = resource_info[columns].iloc[0].to_dict()
                    uri_row['cluster_label'] = new_cluster
                    uri_row['Derived Phrase'] = new_name
                    uri_row = pd.Series(uri_row)
                    uri_row = uri_row.reindex(uri_df.columns)
                    uri_df.loc[len(uri_df)] = uri_row.values
                    uri_df.drop_duplicates(subset=['Resource'], inplace=True)
                    uri_df.reset_index(drop=True, inplace=True)
                    feat_uri_row = pd.DataFrame({'Feature_ID': [feat_id],  'URI' : [new_uri]})
                    feat_uri_df = pd.concat([feat_uri_df, feat_uri_row], ignore_index=True)
                    feat_uri_df.drop_duplicates(subset=['Feature_ID', 'URI'], inplace=True)
                    feat_uri_df.reset_index(drop=True, inplace=True)


        current_clusters = feat_cluster_df['Cluster_ID'].unique()
        current_clusters = current_clusters[~np.isnan(current_clusters)].astype(np.int64)

        new_clusters = [cl for cl in features_clustered['cluster_label'].unique() if cl not in current_clusters]

        exist_feat_df = features_clustered[features_clustered['cluster_label'].apply(lambda x: x in current_clusters)].copy()
        new_feat_df = features_clustered[features_clustered['cluster_label'].apply(lambda x: x in new_clusters)].copy()

        exist_feat_df.reset_index(drop=True, inplace=True)
        new_feat_df.reset_index(drop=True, inplace=True)


    # Update Hierarchy
    print("\nUpdating Hierarchy...")
    feature_cluster_df = feat_cluster_df.dropna(subset=['Cluster_ID']).reset_index().groupby('Feature_ID').agg({
        'Cluster_ID' : list
    }).reset_index()
    current_child_nodes = [node for node in nodes_df['Feature_ID'].unique() if node in hierarchy_df['Child_ID'].unique()]
    current_child_nodes = [node for node in current_child_nodes if node in feature_cluster_df['Feature_ID'].unique()]

    current_orphan_nodes = [node for node in nodes_df['Feature_ID'].unique() if node not in hierarchy_df['Child_ID'].unique()]
    current_orphan_nodes = [node for node in current_orphan_nodes if node != 'feature_Cloud_computing']

    
    while len(current_orphan_nodes)>0:
        # print(f'\n{len(current_orphan_nodes)} orphan nodes left')
        node_similarity_scores = {}
        for node in current_child_nodes:
            node_clusters = feature_cluster_df.loc[(feature_cluster_df['Feature_ID']==node), 'Cluster_ID'].item()
            # print(node_clusters)
            node_embeddings = features_clustered[features_clustered['cluster_label'].apply(lambda x: x in node_clusters)]['embeddings'].to_list()
            if not len(node_embeddings):
                continue
            avg_node_similarities = []

            for orphan_node in current_orphan_nodes:
                orphan_node_clusters = feat_cluster_df[feat_cluster_df['Feature_ID'] == orphan_node]['Cluster_ID'].tolist()
                orphan_node_embeddings = features_clustered[features_clustered['cluster_label'].apply(lambda x: x in orphan_node_clusters)]['embeddings'].to_list()

                if orphan_node_embeddings:
                    # Calculate cosine similarities between node embeddings and existing embeddings
                    cosine_similarities = cosine_similarity(node_embeddings, orphan_node_embeddings)

                    # Compute the max similarity for each embedding in the node list
                    avg_similarities = np.mean(cosine_similarities, axis=1)

                    # Compute the overall average of these average cosine similarities
                    overall_avg_similarity = np.mean(avg_similarities)
                    avg_node_similarities.append(overall_avg_similarity)
                else:
                    avg_node_similarities.append(0)

            # Find the index of the highest average similarity
            max_index = avg_node_similarities.index(max(avg_node_similarities))
            max_value = avg_node_similarities[max_index]

            child_id = current_orphan_nodes[max_index]
            node_similarity_scores[(node, child_id)] = max_value

        top_matches = sorted(node_similarity_scores.items(), key=lambda x: x[1], reverse=True)[:min(10, len(current_orphan_nodes))]

        # Dictionary to keep potential parent-child pairs and their similarity scores
        potential_parents = {}

        # Iterate over top matches and determine the best parent for each child based on hierarchy depth
        for (parent_id, child_id), similarity_score in top_matches:
            if child_id not in potential_parents or potential_parents[child_id][1] > get_node_depth(parent_id, hierarchy_df):
                potential_parents[child_id] = (parent_id, get_node_depth(parent_id, hierarchy_df), similarity_score)

        # Update hierarchy with the chosen parent-child pairs
        for child_id, (parent_id, _, similarity_score) in potential_parents.items():
            # print(f"Finalizing hierarchy update: Parent {parent_id}, Child {child_id}, Similarity {similarity_score}")
            # Add your hierarchy update logic here
            hier_row = pd.DataFrame({'Parent_ID':[parent_id], 'Child_ID':[child_id]})
            hierarchy_df = pd.concat([hierarchy_df, hier_row], ignore_index=True)
            hierarchy_df.drop_duplicates(subset=['Parent_ID', 'Child_ID'], inplace=True)
            hierarchy_df.reset_index(drop=True, inplace=True)
            
        current_child_nodes = [node for node in nodes_df['Feature_ID'].unique() if node in hierarchy_df['Child_ID'].unique()]
        current_child_nodes = [node for node in current_child_nodes if node in feature_cluster_df['Feature_ID'].unique()]

        # handling orphan only nodes.
        current_orphan_nodes = [node for node in nodes_df['Feature_ID'].unique() if node not in hierarchy_df['Child_ID'].unique()]
        current_orphan_nodes = [node for node in current_orphan_nodes if node != 'feature_Cloud_computing']

    nodes_df.to_csv('neo4j/updated/nodes.csv', index=False)
    hierarchy_df.to_csv('neo4j/updated/hierarchy.csv', index=False)
    cluster_df.to_csv('neo4j/updated/cluster_data.csv', index=False)
    feat_cluster_df.to_csv('neo4j/updated/feat_cluster_data.csv', index=False)
    uri_df.drop_duplicates(subset=['Resource', 'cluster_label'], inplace=True, keep='first')
    uri_df.reset_index(drop=True, inplace=True)
    uri_df.to_csv('neo4j/updated/uri_data.csv', index=False)
    uri_df.to_pickle('neo4j/updated/uri_data')
    feat_uri_df.to_csv('neo4j/updated/feat_uri_data.csv', index=False)

    # cluster_df = pd.read_csv('neo4j/updated/cluster_data.csv')
    update_kg_ip_data(cluster_df)

            

if __name__ == '__main__':
    main()