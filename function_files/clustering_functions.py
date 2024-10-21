import numpy as np
import pickle
import re
from nltk import everygrams
from collections import Counter, OrderedDict

def encode_phrases(phrases, model):
    """ Produces phrase embeddings (vectors) """
    embeddings = model.encode(phrases)
    return embeddings  # Directly return the numpy array

def save_clusterer(clusterer):
    """ Saves the clusterer in a pickle file """
    with open('cluster_data/clusterer', 'wb') as file:
        pickle.dump(clusterer, file)

def load_clusterer():
    """ Loads the clusterer from a pickle file """
    """Calculate the cosine similarity between two vectors """
    with open('cluster_data/clusterer', 'rb') as file:
        clusterer = pickle.load(file)
    return clusterer

def predict_cluster_dbpedia(row, clusterer, pca, reduced_embeddings):
    """
    Predicts the cluster label for a given row using pre-trained clustering and PCA models.
    The function reduces the dimensionality of the input embedding using PCA and finds the closest 
    embedding from the reduced embedding space to assign a cluster label.

    Parameters:
        row (pandas.Series): The row containing an embedding to predict the cluster.
        clusterer (object): The clustering model (hdbscan) used to predict clusters.
        pca (object): The PCA model used to reduce dimensionality.
        reduced_embeddings (numpy.ndarray): The PCA-reduced embeddings used for clustering.

    Returns:
        int: The cluster label for the given embedding.
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

def cos_sim(vec1, vec2):
    """Calculate the cosine similarity between two vectors """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 > 0 and norm_vec2 > 0:
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity
    else:
        return 0.0  # If the norm of any vector is zero

# Function to calculate cosine similarities and find maximum similarity
def find_most_similar(embedding, all_embeddings):
    """
    Finds the most similar embedding from a list of embeddings using cosine similarity.

    Parameters:
        embedding (numpy.ndarray): The embedding to compare.
        all_embeddings (list of numpy.ndarray): A list of embeddings to compare against.

    Returns:
        tuple: A tuple containing the index of the most similar embedding and the maximum similarity score.
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

def predict_phrase_cluster(phrase, clusterer, model, pca, reduced_embeddings):
    """
    Predicts the cluster label for a given phrase using a pre-trained clustering and PCA model.
    The function generates an embedding for the phrase, reduces its dimensionality, and finds the 
    closest embedding from the reduced embedding space to assign a cluster label.

    Parameters:
        phrase (str): The phrase to predict the cluster for.
        clusterer (object): The clustering model used for prediction.
        model (object): The embedding model used to encode the phrase.
        pca (object): The PCA model used to reduce dimensionality.
        reduced_embeddings (numpy.ndarray): The PCA-reduced embeddings used for clustering.

    Returns:
        int: The predicted cluster label for the phrase.
    """
    # Generate embedding for the new phrase
    new_embedding = model.encode([phrase])

    # Reduce the dimensionality of the new embedding using the loaded PCA model
    new_embedding_reduced = pca.transform(new_embedding)

    # Assuming 'reduced_embeddings' is loaded and contains the PCA-reduced embeddings used for clustering
    distances = np.linalg.norm(reduced_embeddings - new_embedding_reduced, axis=1)

    # Find the index of the closest embedding
    nearest_sample_index = np.argmin(distances)

    # Return the cluster label of the closest point
    return clusterer.labels_[nearest_sample_index]

def preprocess_phrase(phrase):
    """
    Preprocesses a phrase by replacing hyphens with spaces and temporarily replacing spaces inside parentheses 
    with underscores to preserve them during further processing.

    Parameters:
        phrase (str): The phrase to preprocess.

    Returns:
        str: The preprocessed phrase.
    """
    # Replace hyphens with spaces for uniformity
    phrase = phrase.replace('-', ' ')
    # Temporary replace spaces within parentheses
    return re.sub(r'\(\s*([^)]*?)\s*\)', lambda m: '(' + m.group(1).replace(' ', '_') + ')', phrase)

def postprocess_tokens(tokens):
    """
    Postprocesses a list of tokens by replacing underscores with spaces to restore the original token formatting.

    Parameters:
        tokens (list): A list of tokens to postprocess.

    Returns:
        list: The postprocessed tokens.
    """
    # Replace back the placeholders to spaces
    return [token.replace('_', ' ') for token in tokens]

def compute_ngrams(row, n_min, n_max, min_freq=2):
    """
    Computes n-grams for the processed phrases in a row.
    The function generates n-grams for each phrase, counts their occurrences, and returns 
    only those n-grams that appear with a frequency greater than or equal to a specified threshold.

    Parameters:
        row (pandas.Series): The row containing processed phrases for n-gram extraction.
        n_min (int): The minimum length of n-grams.
        n_max (int): The maximum length of n-grams.
        min_freq (int): The minimum frequency for an n-gram to be included (default is 2).

    Returns:
        dict: A dictionary of filtered n-grams and their counts.
    """

    doc_list = [preprocess_phrase(phrase.lower()) for phrase in row['Processed Phrases']]
    overall_ngram_counts = Counter()
    
    for phrase in doc_list:
        words = phrase.split()
        words = postprocess_tokens(words)  # Post-process tokens to restore original spaces within parentheses
        ngrams = everygrams(words, min_len=n_min, max_len=n_max)
        ngram_counts = Counter(ngrams)
        overall_ngram_counts += ngram_counts

    sorted_ngrams = OrderedDict(sorted(overall_ngram_counts.items(), key=lambda item: item[1], reverse=True))
    filtered_ngrams = {ngram: count for ngram, count in sorted_ngrams.items() if count >= min_freq}

    return filtered_ngrams

def derive_most_representative_ngram(row):
    """
    Derives the most representative n-gram from a row's n-grams by evaluating alternative phrases.
    The function selects the most frequent n-grams and filters them based on their appearance in 
    multiple alternatives, prioritizing those that appear in a significant percentage of alternatives.

    Parameters:
        row (pandas.Series): The row containing n-grams and alternative phrases.

    Returns:
        str: The most representative phrase based on the n-grams, or None if none is found.
    """
    ngrams_dict = row['N-grams']
    alternatives = row['Alternatives']
    # processed_phrases = row['Processed Phrases']
    processed_phrases = [preprocess_phrase(phrase) for phrase in row['Processed Phrases']]

    total_alt = len(set(alternatives))

    # Mapping phrases to sets of alternatives
    phrase_to_alternative = {}
    for phrase, alt in zip(processed_phrases, alternatives):
        if phrase not in phrase_to_alternative:
            phrase_to_alternative[phrase] = set()
        phrase_to_alternative[phrase].add(alt)

    current_ngrams = {n: [] for n in range(1, 7)}
    max_counts = {n: 0 for n in range(1, 7)}

    for ngram, count in ngrams_dict.items():
        n = len(ngram)
        if n <= 5:
            if count > max_counts[n]:
                max_counts[n] = count
                current_ngrams[n] = [ngram]
            elif count == max_counts[n]:
                current_ngrams[n].append(ngram)

    for n in range(1, 5):
        next_level_candidates = []
        for next_ngram in current_ngrams[n+1]:
            for current_ngram in current_ngrams[n]:
                if all(word in next_ngram for word in current_ngram):
                    new_terms = set(next_ngram) - set(current_ngram)
                    vendors = set()
                    for term in new_terms:
                        for phrase in processed_phrases:
                            if term in phrase.lower():
                                vendors.update(phrase_to_alternative[phrase])

                    # Check if new terms appear in phrases from more than 45% of unique vendors
                    if len(vendors) > (0.45 * total_alt):
                        next_level_candidates.append(next_ngram)

        if next_level_candidates:
            current_ngrams[n+1] = next_level_candidates
        else:
            current_ngrams[n+1] = []

    for n in range(5, 0, -1):
        if current_ngrams[n]:
            result = ' '.join(current_ngrams[n][0])
            for phrase in row['Processed Phrases']:
                if result == phrase.lower():
                    return phrase
            for phrase in processed_phrases:
                if result == phrase.lower():
                    return phrase
            return result
    return None
