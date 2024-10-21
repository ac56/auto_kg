import hashlib
import pandas as pd
import os

# Function to create a unique hash
def create_hash(row):
    """
    Creates a unique hash for a given row by combining the cluster label, URL, and original phrase.

    Parameters:
        row (pandas.Series): A row containing 'cluster_label', 'URL', and 'Original Phrase'.

    Returns:
        str: A unique MD5 hash string.
    """
    unique_string = f"{row['cluster_label']}_{row['URL']}_{row['Original Phrase']}"
    return hashlib.md5(unique_string.encode()).hexdigest()


def update_kg_ip_data(cluster_df):
    """
    Updates the knowledge graph information point data by filtering and creating unique IDs.

    Parameters:
        cluster_df (pandas.DataFrame): The DataFrame containing cluster information.

    Returns:
        None: The function updates and saves the information point data to CSV and pickle formats.
    """
    kg_clusters = cluster_df['cluster_label'].unique()
    ip_data = pd.read_pickle('cluster_data/phrase_data')
    kg_ip_data = ip_data[ip_data['cluster_label'].apply(lambda x: x in kg_clusters)].copy()
    kg_ip_data = kg_ip_data[['Alternative', 'URL', 'Specific Content', 'Original Phrase', 'cluster_label']]
    # Create unique IDs for information points
    kg_ip_data['InfoPoint_ID'] = kg_ip_data.apply(create_hash, axis=1)
    kg_ip_data.reset_index(drop=True, inplace=True)

    pickle_file = 'neo4j/kg_ip_data'
    if os.path.exists(pickle_file):
        kg_ip_data.to_csv('neo4j/updated/kg_ip_data.csv', index=False)
        kg_ip_data.to_pickle('neo4j/updated/kg_ip_data')
    else:
        kg_ip_data.to_csv('neo4j/kg_ip_data.csv', index=False)
        kg_ip_data.to_pickle('neo4j/kg_ip_data')

def get_node_depth(node, hierarchy_df):
    """
    Calculates the depth of a node in a hierarchy by recursively traversing its parent nodes.

    Parameters:
        node (str): The node whose depth is to be calculated.
        hierarchy_df (pandas.DataFrame): The DataFrame containing the hierarchy information.

    Returns:
        int: The depth of the node in the hierarchy.
    """
    depth = 0
    current_node = node
    while True:
        if current_node not in hierarchy_df['Child_ID'].values:
            break
        current_node = hierarchy_df[hierarchy_df['Child_ID'] == current_node]['Parent_ID'].iloc[0]
        depth += 1
    return depth

def get_uri_info(uri, df, get_all):
    """
    Retrieves detailed information about a given URI from a DataFrame, with an option to include all columns or a subset.

    Parameters:
        uri (str): The URI to look up.
        df (pandas.DataFrame): The DataFrame containing resource information.
        get_all (bool): If True, return all columns; otherwise, return a subset of 16 columns.

    Returns:
        pandas.Series: A Series containing the requested information for the URI.
    """
    # Find the row corresponding to the given URI
    row = df[df['Resource'] == uri]
    
    # Handle the case where the row is not found
    if row.empty:
        # Return a Series with 16 None values, matching the required columns
        return pd.Series([None]*16, index=['Genre(s)', 'GenreNames', 'GenreOf', 'GenreOfNames', 'dbo_type(s)', 'Developer(s)', 'Website(s)', 'Disambiguate(s)', 'DisambiguateOf', 'DisambiguateOfNames', 'Product(s)', 'ProductNames', 'ProductOf', 'ProductOfNames', 'LinkedResources', 'LinkedNames'])
    
    # Check if there are redirects
    redirect = row['Redirects'].values[0]  # Get the redirect value
    
    # If there is a redirect, use the first URI in the redirects
    if redirect:
        uri = redirect[0]
        row = df[df['Resource'] == uri]
        
        # If redirected row is not found, return None series
        if row.empty:
            return pd.Series([None]*16, index=['Genre(s)', 'GenreNames', 'GenreOf', 'GenreOfNames', 'dbo_type(s)', 'Developer(s)', 'Website(s)', 'Disambiguate(s)', 'DisambiguateOf', 'DisambiguateOfNames', 'Product(s)', 'ProductNames', 'ProductOf', 'ProductOfNames', 'LinkedResources', 'LinkedNames'])
    
    if get_all:
        return row[['Resource', 'Label', 'Abstract', 'Genre(s)', 'GenreNames', 'GenreOf', 'GenreOfNames', 'dbo_type(s)', 'Developer(s)', 'Website(s)', 'Disambiguate(s)', 'DisambiguateOf', 'DisambiguateOfNames', 'Product(s)', 'ProductNames', 'ProductOf', 'ProductOfNames', 'LinkedResources', 'LinkedNames']].iloc[0]
    else:
        # Return only the relevant 16 columns as a Series
        return row[['Genre(s)', 'GenreNames', 'GenreOf', 'GenreOfNames', 'dbo_type(s)', 'Developer(s)', 'Website(s)', 'Disambiguate(s)', 'DisambiguateOf', 'DisambiguateOfNames', 'Product(s)', 'ProductNames', 'ProductOf', 'ProductOfNames', 'LinkedResources', 'LinkedNames']].iloc[0]