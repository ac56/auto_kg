import re
import string
import pandas as pd
import numpy as np

def normalize_text(input_string):
    """
    Normalizes text by removing punctuation, extra spaces, and non-English characters.

    Args:
        input_string (str): The input text string.

    Returns:
        str: The normalized text string.
    """
    space_pattern = r'\u00A0|\u200B|\u2002|\u2003|\u2004|\xa0|\u200b'
    # Remove punctuation using regular expression
    cleaned_string = re.sub(r'[^\w\s]', '', input_string)
    # Remove extra spaces
    cleaned_string = re.sub(space_pattern, ' ', cleaned_string)
    cleaned_string = re.sub(r'\s+', '', cleaned_string)
    # Strip leading and trailing spaces
    cleaned_string = cleaned_string.strip()
    return cleaned_string


def remove_parentheses(text):
    """
    Removes text within parentheses from the input string.

    Args:
        text (str): The input text string.

    Returns:
        str: The text string with parentheses and their contents removed.
    """
    # Use regular expression to find and remove text inside parentheses
    return re.sub(r'\s*\(.*?\)\s*', ' ', text).strip()

def remove_non_english_characters(text):
    """
    Removes non-English characters from the input text string.

    Args:
        text (str): The input text string.

    Returns:
        str: The text string with non-English characters removed.
    """
    # Define the allowed characters: English letters, numbers, punctuation, and whitespace
    allowed_chars = string.ascii_letters + string.digits + string.punctuation + ' \t\n\r\x0b\x0c'
    # Create a regex pattern to match any character not in the allowed set
    pattern = f'[^{re.escape(allowed_chars)}]'
    # Use re.sub() to replace non-English characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def get_sentences(text, segmenter):
    """
    Splits the input text into sentences using a sentence segmentation model.

    Handles empty strings, whitespace-only strings, and newlines within sentences.

    Args:
        text (str): The input text string.
        segmenter (object): A sentence segmentation model.

    Returns:
        list: A list of extracted sentences.
    """
    # Check for empty or whitespace-only strings
    if not text.strip():
        return []  # Return an empty list if text is empty or contains only whitespace
    
    # Define pattern to split sentences into parts using new lines
    pattern = r'\n\s*\n*'  # Matches one or more new lines with optional whitespace
    sents = segmenter.segment(text)

    # Split text into sentences
    sentences = []
    for sent in sents:
        sent_text = sent.strip()
        if sent_text:
            # Split each sentence into parts using the defined pattern
            parts = re.split(pattern, sent_text)
            # Add non-empty parts to the list of sentences
            sentences.extend(part.strip() for part in parts if part)
    
    return sentences

def is_nan_none_or_empty(x):
    """
    Checks if the input value is NaN, None, or empty.

    Handles lists, NumPy arrays, and other data types.

    Args:
        x: The value to check.

    Returns:
        bool: True if the value is NaN, None, or empty; otherwise, False.
    """
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return len(x) == 0
    return pd.isna(x)

def choose_contexts(row, n):
    """
    Chooses n relevant contexts from a given row of data.

    Prioritizes contexts that contain the derived phrase or related phrases.
    If no matching contexts are found, uses all contexts.
    Selects the median context and expands outwards to include n contexts.

    Args:
        row (dict): A dictionary containing data for the prompt.
        n (int): The number of contexts to select.

    Returns:
        list: A list of selected contexts.
    """
    contexts = list(set(row['Contexts']))
    derived_phrase = row['Derived Phrase']
    proc_phrase = remove_parentheses(normalize_text(derived_phrase.replace('-', ' ').lower()))
    norm_context_list = [remove_parentheses(normalize_text(context.replace('-', ' ').lower())) for context in contexts]

    # Filter contexts that contain the derived phrase
    filtered_contexts = []
    lengths = []
    
    for i, norm_context in enumerate(norm_context_list):
        if proc_phrase in norm_context or any(phrase in norm_context for phrase in proc_phrase.split()):
            filtered_contexts.append(contexts[i])
            lengths.append(len(norm_context.split()))  # Measure word count of each context

    # If no contexts match the phrase, use all contexts for median calculation
    if not filtered_contexts:
        # Calculate lengths of all contexts
        lengths = [len(norm_context.split()) for norm_context in norm_context_list]
        # Use all contexts since none contain the phrase
        filtered_contexts = contexts

    # Sort the lengths and retain original indices for reference
    sorted_lengths = sorted((length, i) for i, length in enumerate(lengths))
    
    # Extract the sorted lengths and the corresponding original indices
    sorted_length_values = [length for length, _ in sorted_lengths]
    sorted_indices = [i for _, i in sorted_lengths]
    
    # Find the median index
    median_index = len(sorted_length_values) // 2  # Median is the middle index in the sorted list

    # Start with the median context
    selected_indices = [median_index]
    
    # Expand to n-1 additional contexts by moving outwards from the median, prioritizing smaller lengths
    left = median_index - 1
    right = median_index + 1
    while len(selected_indices) < n:
        if left >= 0 and (right >= len(sorted_length_values) or (right >= len(sorted_length_values)) or (sorted_length_values[median_index] - sorted_length_values[left] <= sorted_length_values[right] - sorted_length_values[median_index])):
            selected_indices.append(left)
            left -= 1
        elif right < len(sorted_length_values):
            selected_indices.append(right)
            right += 1
        else:
            break

    # Retrieve the selected contexts from the filtered list based on the sorted indices
    selected_contexts = [filtered_contexts[sorted_indices[i]] for i in selected_indices]
    
    return selected_contexts