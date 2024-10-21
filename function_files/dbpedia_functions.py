import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from numpy.linalg import norm
from rdflib import Graph
from rdflib.namespace import RDF, RDFS
from rdflib import URIRef, Namespace
from SPARQLWrapper import JSON, SPARQLWrapper
from urllib.error import HTTPError
from urllib.parse import quote


def get_concept_list(df):
    """
    Generates a list of unique concepts by combining 'Concept' and 'NarrowerConcept' columns from a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing concept information.

    Returns:
        list: A list of unique concept names.
    """
    all_concepts = pd.concat([df['Concept'], df['NarrowerConcept']]).unique()
    concept_list = []
    for concept_uri in all_concepts:
        # Split the URL by '/'
        parts = concept_uri.split('/')
        # Get the part containing the category name
        category_part = parts[-1]
        # Split the category part by ':'
        category_parts = category_part.split(':')
        # Get the concept name
        concept_list.append(category_parts[-1])

    return concept_list

def get_resource_list(df):
    """
    Generates a list of unique resources by combining 'SubjectOf' and 'LinkedFrom' columns from a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing resource information.

    Returns:
        list: A list of unique resource names.
    """
    all_resources = list(set().union(*df['SubjectOf'], *df['LinkedFrom']))
    resource_list = []
    for resource_uri in all_resources:
        # Split the URL by '/'
        parts = resource_uri.split('/')
        # Get the part containing the category name
        resource_list.append(parts[-1])

    return resource_list

def get_concept_name(uri):
    """
    Extracts the concept name from a URI by removing prefixes and replacing underscores with spaces.

    Parameters:
        uri (str): The URI to extract the concept name from.

    Returns:
        str: The cleaned concept name.
    """
    # Split the URI on '/' and get the last part
    concept_name = uri.split('/')[-1]
    # Remove the 'Category:' prefix if present
    if concept_name.startswith('Category:'):
        concept_name = concept_name[len('Category:'):]
    # Replace '_' with ' ' in the concept name
    return concept_name.replace('_', ' ')

def correct_name(name):
    """
    Corrects a name by removing the 'Category:' prefix if present.

    Parameters:
        name (str): The name to be corrected.

    Returns:
        str: The corrected name without the 'Category:' prefix.
    """
    name = str(name)
    if name.startswith('Category:'):
        name = name[len('Category:'):]
    return name

def get_resource_name(uri):
    """
    Extracts and cleans the resource name from a URI by replacing underscores with spaces and applying corrections.

    Parameters:
        uri (str): The URI to extract the resource name from.

    Returns:
        str: The cleaned resource name.
    """
    # Split the URI on '/' and get the last part
    resource_name = uri.split('/')[-1]
    # Replace '_' with ' ' in the resource name
    resource_name = resource_name.replace('_', ' ')
    return correct_name(resource_name)

def get_resource_names(resource_info):
    """
    Generates resource names for specific columns in the resource information DataFrame by applying the `get_resource_name` function.

    Parameters:
        resource_info (pandas.DataFrame): The DataFrame containing resource information.

    Returns:
        pandas.DataFrame: The updated DataFrame with extracted resource names.
    """
    # # Apply the function to each URI in the lists of the 'LinkedResources' column
    resource_info['LinkedNames'] = resource_info['LinkedResources'].apply(lambda uris: [get_resource_name(uri) for uri in uris] if uris else None)
    resource_info['GenreNames'] = resource_info['Genre(s)'].apply(lambda uris: [get_resource_name(uri) for uri in uris] if uris else None)
    resource_info['GenreOfNames'] = resource_info['GenreOf'].apply(lambda uris: [get_resource_name(uri) for uri in uris] if uris else None)
    resource_info['DisambiguateOfNames'] = resource_info['DisambiguateOf'].apply(lambda uris: [get_resource_name(uri) for uri in uris] if uris else None)
    resource_info['ProductNames'] = resource_info['Product(s)'].apply(lambda uris: [get_resource_name(uri) for uri in uris] if uris else None)
    resource_info['ProductOfNames'] = resource_info['ProductOf'].apply(lambda uris: [get_resource_name(uri) for uri in uris] if uris else None)

    resource_order = ['Resource', 'Label', 'Abstract', 'Comment', 'Genre(s)', 'GenreNames','GenreOf', 
    'GenreOfNames', 'dbo_type(s)', 'Developer(s)', 'Website(s)', 'Disambiguate(s)',
    'DisambiguateOf', 'DisambiguateOfNames', 'Product(s)', 'ProductNames', 'ProductOf', 'ProductOfNames', 'LinkedResources',
    'LinkedNames', 'Redirects']
    resource_info = resource_info[resource_order]
    return resource_info

    
def update_abstracts(row, resource_info):
    """
    Updates the abstract for a row by looking up resource information, filling in missing abstracts.

    Parameters:
        row (pandas.Series): The row containing abstract and URI information.
        resource_info (pandas.DataFrame): The DataFrame containing resource information.

    Returns:
        str: The updated abstract, or 'No abstract found' if none is available.
    """
    abstract = row['Abstracts']
    if abstract in ['TBD', 'No Abstract']:
        uri = row['URIs']
        # Lookup the matching row(s) in resource_info based on URI
        info_rows = resource_info[resource_info['Resource'] == uri]
        if not info_rows.empty:
            # Check if all Abstract values are NaN
            if info_rows['Abstract'].isna().all():
                if info_rows['Redirects'].notna().any() and len(info_rows['Redirects'].values[0]) > 0:
                    redirect = info_rows['Redirects'].values[0][0]  # Safely access the first redirect
                    re_info_rows = resource_info[resource_info['Resource'] == redirect]
                    if not re_info_rows.empty:
                        if re_info_rows['Abstract'].isna().all():
                            return 'No abstract available'
                        else:
                            new_abstract = re_info_rows['Abstract'].dropna().iloc[0]
                            return new_abstract
                    else:
                        return 'No abstract available'
                else:
                    return 'No abstract available'
            else:
                new_abstract = info_rows['Abstract'].dropna().iloc[0]
                return new_abstract
        else:
            # If there are no matching rows, keep as 'TBD'
            return 'No abstract found'
    else:
        return abstract


def get_abstract(uri, resource_info):
    """ Returns the abstract of the Resource with the specified URI, from the Dataframe containing the resources information  """
    # Lookup the matching row(s) in resource_info based on URI
    row = resource_info[resource_info['Resource'] == uri]

    if row.empty:
        return None

    return row['Abstract'].iloc[0]

def construct_query_for_narrower_concepts(concept_uri):
    """
    Constructs a SPARQL query to find narrower concepts for a given concept URI.
    
    Parameters:
    - concept_uri: The URI of the concept for which to find narrower concepts.
    
    Returns:
    A string representing the SPARQL query.
    """
    query = f"""
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    
    SELECT ?narrowerConcept WHERE {{
      ?narrowerConcept skos:broader <{concept_uri}> .
    }}
    """
    return query

def construct_query_for_subject_of(concept_uri):
    """
    Constructs a SPARQL query to find resources for which a given concept URI is a subject.
    
    Parameters:
    - concept_uri: The URI of the concept for which to find resources.
    
    Returns:
    A string representing the SPARQL query.
    """
    query = f"""
    PREFIX dcterms: <http://purl.org/dc/terms/>
    SELECT ?resource WHERE {{
      ?resource dcterms:subject <{concept_uri}> .
    }}
    """
    return query

def construct_query_for_linked_from(concept_uri):
    """
    Constructs a SPARQL query to find resources linked to the concept specified.
    
    Parameters:
    - concept_uri: The URI of the concept for which to find linked resources.
    
    Returns:
    A string representing the SPARQL query.
    """
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT ?resource WHERE {{
      ?resource dbo:wikiPageWikiLink <{concept_uri}> .
    }}
    """
    return query

def construct_query_for_product_of(resource_uri):
    """
    Constructs a SPARQL query to find resources to which the specified resource is a product of.
    
    Parameters:
    - resource_uri: The URI of the resource which is product to the searched resources.
    
    Returns:
    A string representing the SPARQL query.
    """
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbp: <http://dbpedia.org/property/>
    SELECT DISTINCT ?resource WHERE {{
      {{ ?resource dbo:product <{resource_uri}> . }}
      UNION
      {{ ?resource dbp:product <{resource_uri}> . }}
    }}
    """
    return query

def construct_query_for_genre_of(resource_uri):
    """
    Constructs a SPARQL query to find resources to which the specified resource is a genre of.
    
    Parameters:
    - resource_uri: The URI of the resource which is genre to the searched resources.
    
    Returns:
    A string representing the SPARQL query.
    """
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbp: <http://dbpedia.org/property/>
    SELECT DISTINCT ?resource WHERE {{
      {{ ?resource dbo:genre <{resource_uri}> . }}
      UNION
      {{ ?resource dbp:genre <{resource_uri}> . }}
    }}
    """
    return query

def construct_query_for_disamb(resource_uri):
    """
    Constructs a SPARQL query to find resources which the specified resource disambiguates.
    
    Parameters:
    - resource_uri: The URI of the resource which disambiguates the searched resources.
    
    Returns:
    A string representing the SPARQL query.
    """
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbp: <http://dbpedia.org/property/>
    SELECT DISTINCT ?resource WHERE {{
      {{ ?resource dbo:wikiPageDisambiguates <{resource_uri}> . }}
      UNION
      {{ ?resource dbo:wikiPageDisambiguates <{resource_uri}> . }}
    }}
    """
    return query

def run_sparql_query(query):
    """
    Runs a SPARQL query against the DBpedia endpoint and returns the results.
    """
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    return results

def encode_uri(uri):
    # Encode special characters in the URI, excluding '/', ':', and other safe characters
    return quote(uri, safe="/:")


def run_sparql_query_narrow(query):
    """
    Runs a SPARQL query against the DBpedia endpoint and returns the results.
    """
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    return [result["narrowerConcept"]["value"] for result in results["results"]["bindings"]]

def process_results(results):
    """
    Process SPARQL query results to extract a list of URI values.

    Args:
    - results (dict): The SPARQL query results.

    Returns:
    - list: A list of extracted URI values.
    """
    uri_list = []
    for binding in results.get('results', {}).get('bindings', []):
        resource_uri = binding.get('resource', {}).get('value')
        if resource_uri:
            uri_list.append(resource_uri)
    return uri_list

def explore_narrower_concepts(concept_uri, explored, dataframe):
    """
    Recursively explores narrower concepts starting from a given concept URI and stores results in a DataFrame.
    """
    if concept_uri in explored:
        return dataframe
    explored.add(concept_uri)
    narrower_concepts = run_sparql_query_narrow(construct_query_for_narrower_concepts(concept_uri))
    
    for narrower in narrower_concepts:
        # Append to DataFrame
        new_row = pd.DataFrame({'Concept': [concept_uri], 'NarrowerConcept': [narrower]})
        dataframe = pd.concat([dataframe, new_row], ignore_index=True)
        dataframe = explore_narrower_concepts(narrower, explored, dataframe)
    
    return dataframe

def get_redirects(uri, dbo):
    """
    Locates the DBPedia page to which the uri redirects to, if any.
    """
    # dbo = Namespace("http://dbpedia.org/ontology/")
    encoded_uri = encode_uri(uri)

    g = Graph()
    try:
        g.parse(encoded_uri)
        print(f"Successfully parsed {uri}")
    except HTTPError as e:
        print(f"Failed to parse {uri}: {e}")
    except Exception as e:
        print(f"An error occurred with {uri}: {e}")

    redirects = None
    redirects = list(g.objects(predicate=dbo.wikiPageRedirects))
    redirects_list = [str(uri) for uri in list(redirects)]

    return redirects_list

def create_resource_dataframe(resource_list):
    """
    Creates the Dataframe storing all the DBpedia resources information.
    """
    dbo = Namespace("http://dbpedia.org/ontology/")
    dbp = Namespace("http://dbpedia.org/property/")
    resource_df = pd.DataFrame(columns=['Resource', 'Label', 'Abstract', 'Comment', 'Genre(s)', 'GenreOf', 'dbo_type(s)', 'Developer(s)', 'Website(s)', 'Disambiguate(s)', 'DisambiguateOf', 'Product(s)', 'ProductOf', 'LinkedResources', 'Redirects'])
    failed_to_parse = []
    failed_to_query = []
    for uri in tqdm(resource_list, desc="Processing URIs", unit="uri"):
        encoded_uri = encode_uri(uri)
        subject_uri = URIRef(uri)
        g = Graph()
        try:
            g.parse(encoded_uri)
        except HTTPError as e:
            print(f"Failed to parse {uri}: {e}")
            failed_to_parse.append(uri)
            continue
        except Exception as e:
            print(f"An error occurred with {uri}: {e}")
            failed_to_parse.append(uri)
            continue
        # result = g.parse(uri)

        eng_label = None
        labels = list(g.objects(predicate=RDFS.label))
        for label in labels:
            if label.language == "en":
                eng_label = str(label)
                break
        
        eng_comment = None
        comments = list(g.objects(predicate=RDFS.comment))
        for comment in comments:
            if comment.language == "en":
                eng_comment = str(comment)
                break

        abstracts = None
        abstracts = list(g.objects(predicate=dbo.abstract))
        eng_abstract = None
        for abstract in abstracts:
            # Check if the abstract is a literal with a language attribute
            if hasattr(abstract, "language") and abstract.language == "en":
                eng_abstract = str(abstract)
        
        genres = None
        genres = list(g.objects(predicate=dbo.genre))
        if not genres:
            genres = list(g.objects(predicate=dbp.genre))
        genres_list = [str(uri) for uri in list(genres)]
        
        rdf_types = None
        rdf_types = list(g.objects(predicate=RDF.type))
        # Filter results to include only those in the dbo namespace
        dbo_types = [str(t) for t in rdf_types if t.startswith(dbo)]

        developers = None
        developers = list(g.objects(predicate=dbo.developer))
        if not developers:
            developers = list(g.objects(predicate=dbp.developer))
        dev_list = [str(uri) for uri in list(developers)]

        websites = None
        websites = list(g.objects(predicate=dbp.website))
        web_list = [str(uri) for uri in list(websites)]

        disambs = None
        disambs = list(g.objects(predicate=dbo.wikiPageDisambiguates))
        disambs_list = [str(uri) for uri in list(disambs)]

        redirects = None
        redirects = list(g.objects(predicate=dbo.wikiPageRedirects))
        redirects_list = [str(uri) for uri in list(redirects)]

        products = None
        products = list(g.objects(predicate=dbp.product))
        if not products:
            products = list(g.objects(predicate=dbo.product))
        product_list = [str(uri) for uri in list(products)]

        linked_resources = None
        linked_resources = list(g.objects(predicate=dbo.wikiPageWikiLink))
        linked_list = [str(uri) for uri in list(linked_resources)]

        genre_of_list = None
        product_of_list = None 
        disamb_of_list = None
        try:
            genre_of_list = process_results(run_sparql_query(construct_query_for_genre_of(subject_uri)))
            product_of_list = process_results(run_sparql_query(construct_query_for_product_of(subject_uri)))
            disamb_of_list = process_results(run_sparql_query(construct_query_for_disamb(subject_uri)))
        except HTTPError as e:
            print(f"Failed to query {uri}: {e}")
            failed_to_query.append(uri)
        except Exception as e:
            print(f"An unexpected error occurred with quering {uri}: {e}")
            failed_to_query.append(uri)

        new_row = pd.DataFrame({'Resource': [uri], 
                                'Label' : [eng_label],
                                'Abstract': [eng_abstract], 
                                'Comment' : [eng_comment], 
                                'Genre(s)' : [genres_list], 
                                'GenreOf' : [genre_of_list], 
                                'dbo_type(s)' : [dbo_types], 
                                'Developer(s)' : [dev_list], 
                                'Website(s)' : [web_list], 
                                'Disambiguate(s)' : [disambs_list], 
                                'DisambiguateOf' : [disamb_of_list], 
                                'Product(s)' : [product_list], 
                                'ProductOf' : [product_of_list], 
                                'LinkedResources' : [linked_list],
                                'Redirects': [redirects_list]
                                })
        resource_df = pd.concat([resource_df, new_row], ignore_index=True)
    return resource_df, failed_to_parse, failed_to_query

def update_resource_info(new_uris, resource_info):
    """
    Updates the resource information with new URIs by fetching additional details, resolving incomplete data,
    and saving the updated information to CSV and pickle files.

    Parameters:
        new_uris (list): A list of new URIs to fetch information for.
        resource_info (pandas.DataFrame): The existing resource information DataFrame.

    Returns:
        pandas.DataFrame: The updated resource information DataFrame.
    """
    new_uris = [name for name in new_uris if 'Category:' not in name]
    new_resource_info, new_failed_to_parse, new_failed_to_query = create_resource_dataframe(new_uris)
    new_resource_info = get_resource_names(new_resource_info)
    combined_resource_info = pd.concat([resource_info, new_resource_info], ignore_index=True)

    incomplete_info = combined_resource_info[pd.isna(combined_resource_info['Abstract'])]
    extra_names = [item[0] for item in incomplete_info[incomplete_info['LinkedResources'].apply(lambda x: len(x) == 1)]['LinkedResources'] if item[0] not in combined_resource_info['Resource'].to_list()]
    extra_names = list(set(extra_names))

    extra_resource_info, extra_failed_to_parse, extra_failed_to_query = create_resource_dataframe(extra_names)
    extra_resource_info = get_resource_names(extra_resource_info)
    combined_resource_info = pd.concat([combined_resource_info, extra_resource_info], ignore_index=True)

    combined_resource_info.to_csv('dbpedia/resource_info.csv', index=False)
    combined_resource_info.to_pickle('dbpedia/resource_info')

    return combined_resource_info

def get_new_dbpedia_vocab(resource_info, dbpedia_vocab):
    """
    Identifies new DBpedia vocabulary by filtering out resources already present in the existing DBpedia vocabulary.

    Parameters:
        resource_info (pandas.DataFrame): The resource information DataFrame.
        dbpedia_vocab (pandas.DataFrame): The existing DBpedia vocabulary DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame of new DBpedia vocabulary entries.
    """

    # Filter 'combined_resource_info' based on 'Resource' not present in 'URI' column of 'dbpedia_vocab'
    new_dbpedia_vocab = resource_info[~resource_info['Resource'].isin(dbpedia_vocab['URI'])]
    # Select only the columns 'Resource' and 'Label'
    new_dbpedia_vocab = new_dbpedia_vocab[['Resource', 'Label']]
    # Rename columns to match the 'URI' and 'Name' columns of 'dbpedia_vocab'
    new_dbpedia_vocab.columns = ['URI', 'Name']
    # Reset index
    new_dbpedia_vocab.reset_index(drop=True, inplace=True)
    return new_dbpedia_vocab

def explode_and_label(df, uri_col, name_col, type_label):
    """
    Explodes a DataFrame column containing lists into separate rows, adds a 'Type' label to the data,
    and renames columns for consistency.

    Parameters:
        df (pandas.DataFrame): The DataFrame to explode.
        uri_col (str): The column containing URIs.
        name_col (str): The column containing names.
        type_label (str): The label for the 'Type' column.

    Returns:
        pandas.DataFrame: The exploded DataFrame with an additional 'Type' column.
    """
    temp_df = df[[uri_col, name_col]].dropna().explode([uri_col, name_col])
    temp_df['Type'] = type_label
    temp_df.columns = ['URI', 'Name', 'Type']
    return temp_df


def cos_sim(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 > 0 and norm_vec2 > 0:
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity
    else:
        return 0.0  # If the norm of any vector is zero
    
def calc_name_cos_sim(row, column, df, dbpedia_vocab, model, pca):
    """
    Calculates the average cosine similarity between a row's name embedding and a set of phrase embeddings,
    using PCA for dimensionality reduction.

    Parameters:
        row (pandas.Series): The row containing name information and indices.
        column (str): The column containing the name.
        df (pandas.DataFrame): The DataFrame containing embeddings.
        dbpedia_vocab (pandas.DataFrame): The DBpedia vocabulary with embeddings.
        model (object): The embedding model.
        pca (object): The PCA model for dimensionality reduction.

    Returns:
        float: The average cosine similarity between the name and the phrases.
    """
    name = row[column]
    if not pd.isna(name):
        if name in dbpedia_vocab['Name'].unique():
            name_emb = dbpedia_vocab[dbpedia_vocab['Name'] == name].iloc[0]['embeddings']
        else:
            name_emb = model.encode(name)
        if name_emb.ndim == 1:
            name_emb = name_emb.reshape(1, -1)  # Reshape from (N,) to (1, N)
        sub_df = df.iloc[row['Indices']]
        phrase_embs = np.stack(sub_df['embeddings'].values) 

        reduced_name_emb = pca.transform(name_emb)
        reduced_phrase_embs = pca.transform(phrase_embs)
        
        # Calculate the cosine similarity for each phrase with the name
        similarities = [cos_sim(reduced_name_emb, phrase) for phrase in reduced_phrase_embs]

        return np.mean(similarities)
    else:
        return np.nan
    
def append_dbpedia_data(row, sim_threshold):
    """
    Appends DBpedia data to a row based on a similarity threshold.

    Parameters:
        row (pandas.Series): The row containing DBpedia data to append.
        sim_threshold (float): The similarity threshold to consider a match.

    Returns:
        pandas.Series: The updated row with DBpedia data appended.
    """
    if row['Name Avg Cos Sim'] > sim_threshold:
        row['URIs'].append(row['URI'] if pd.notna(row['URI']) else 'No URI')
        row['Names'].append(row['Name'] if pd.notna(row['Name']) else 'No Name')
        row['Abstracts'].append(row['Abstract'] if pd.notna(row['Abstract']) else 'No Abstract')
        row['Names Avg Cos Sim'].append(row['Name Avg Cos Sim'])
    return row

def aggregate_lists(series):
    """
    Aggregates lists from a series of lists into a single list.

    Parameters:
        series (pandas.Series): A pandas Series containing lists.

    Returns:
        list: A single list containing all elements from the series.
    """
    aggregated_list = []
    for item in series:
        aggregated_list.extend(item)
    return aggregated_list


def dbpedia_lookup(keyword):
    """
    Performs a DBpedia lookup for a keyword and returns the top search results.

    Parameters:
        keyword (str): The keyword to search for in DBpedia.

    Returns:
        dict: A dictionary of search results from DBpedia, or an empty dictionary if the request fails.
    """
    url = "http://lookup.dbpedia.org/api/search"
    headers = {"Accept": "application/json"}
    params = {"query": keyword, "maxResults": 5, "format": "json"}
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return {'docs': []}  # Return an empty list if the status code is not 200
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {'docs': []}  # Return an empty list in case of a request failure

def encode_phrases(phrases, model):
    """
    Encodes a list of phrases using a pre-trained embedding model.

    Parameters:
        phrases (list of str): The list of phrases to encode.
        model (object): The embedding model used to generate embeddings.

    Returns:
        numpy.ndarray: The array of embeddings for the phrases.
    """
    embeddings = model.encode(phrases)
    return embeddings  # Directly return the numpy array

def dbpedia_row_search(row, df, model, pca):
    """
    Performs a DBpedia search for a row's derived phrase and calculates cosine similarity with known phrases.

    Parameters:
        row (pandas.Series): The row containing the derived phrase and embedding data.
        df (pandas.DataFrame): The DataFrame containing embeddings.
        model (object): The embedding model used to encode phrases.
        pca (object): The PCA model for dimensionality reduction.

    Returns:
        pandas.Series: A series containing the best-matching URI, name, and similarity score, or NaN if no match is found.
    """
    result = dbpedia_lookup(row['Derived Phrase'])
    result_data = result['docs']  # This accesses the list of documents
    resources = []
    categories = []
    # Loop through each document in the result
    for doc in result_data:
        # Extract the 'resource' values
        if 'resource' in doc:
            resources.extend(doc['resource'])  # Adds all resources from the current document
        
        # Extract the 'category' values
        if 'category' in doc:
            categories.extend(doc['category'])  # Adds all categories from the current document

    resources = resources[:min(5, len(resources))]
    categories = categories[:min(5, len(categories))]
    URIs = resources + categories
    names = [get_resource_name(uri) for uri in URIs]

    # Calculate cosine similarities if possible
    if names:
        name_vectors = list(encode_phrases(names, model))
        sub_df = df.iloc[row['Indices']]
        phrase_embs = np.stack(sub_df['embeddings'].values) 
        reduced_phrase_embs = pca.transform(phrase_embs)
        average_similarities = []

        for name_vec in name_vectors:
            if name_vec.ndim == 1:
                name_vec = name_vec.reshape(1, -1)  # Reshape from (N,) to (1, N)
            name_vec = pca.transform(name_vec)
            if norm(name_vec) > 0:
                # Calculate the cosine similarity for each phrase with the name
                similarities = [cos_sim(name_vec, phrase) for phrase in reduced_phrase_embs]
                average_similarities.append(np.mean(similarities) if similarities else 0)
            else:
                average_similarities.append(0)

        # Handle the case where cosine_similarities is empty or all entries are zero
        if URIs and average_similarities:
            max_index = np.argmax(average_similarities)
            return pd.Series([URIs[max_index], names[max_index], average_similarities[max_index]])
        else:
            return pd.Series([np.nan, np.nan, np.nan])
    else:
        return pd.Series([np.nan, np.nan, np.nan])
    
def predict_cluster_dbpedia(row, clusterer, pca, reduced_embeddings):
    """
    Predicts the cluster label for a given row's embedding by comparing it with pre-trained clusters.
    The function reduces the dimensionality of the embedding using PCA and finds the nearest cluster.

    Parameters:
        row (pandas.Series): The row containing the embedding.
        clusterer (object): The clustering model.
        pca (object): The PCA model for dimensionality reduction.
        reduced_embeddings (numpy.ndarray): The PCA-reduced embeddings used for clustering.

    Returns:
        int: The predicted cluster label.
    """
    embedding = np.array(row['embeddings'])  # Ensure it's a numpy array

    # Check if embedding needs to be reshaped
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)  # Reshape from (N,) to (1, N) if it's a single embedding

    # Reduce the dimensionality of the embedding using the loaded PCA model
    embedding_reduced = pca.transform(embedding)

    # Assuming 'reduced_embeddings' is loaded and contains the PCA-reduced embeddings used for clustering
    distances = np.linalg.norm(reduced_embeddings - embedding_reduced, axis=1)

    # Find the index of the closest embedding
    nearest_sample_index = np.argmin(distances)

    # Return the cluster label of the closest point
    return clusterer.labels_[nearest_sample_index]

# Function to calculate cosine similarities and find maximum similarity
def find_most_similar(embedding, all_embeddings):
    """
    Finds the most similar embedding from a list of embeddings using cosine similarity.

    Parameters:
        embedding (numpy.ndarray): The embedding to compare.
        all_embeddings (list of numpy.ndarray): The list of embeddings to compare against.

    Returns:
        tuple: The index of the most similar embedding and the similarity score.
    """
   # Initialize variables to store the maximum similarity and its index
    max_similarity = -1
    most_similar_index = None
    
    # Compare the current embedding with all embeddings in feature_doc_df
    for idx, feature_embedding in enumerate(all_embeddings):
        similarity = cos_sim(embedding, feature_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_index = idx
    return most_similar_index, max_similarity