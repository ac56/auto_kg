import pandas as pd
import re
from nltk import Tree
from nltk import word_tokenize, ne_chunk
from flair.data import Sentence
from itertools import chain
 


def preprocess_chunks(chunked):
    """
    Preprocesses a list of chunked items (trees or tuples) by filtering out elements where the token is 'vendor'.

    This function iterates through a list of parsed chunks (which may include syntactic trees and token-POS tuples), 
    removing any 'vendor' tokens. It returns a list of preprocessed items, preserving non-'vendor' items.

    Parameters:
        chunked (list): A list of parsed chunks, which may include Tree objects and tuples representing tokens and POS tags.

    Returns:
        list: A list of preprocessed chunks, with 'vendor' tokens removed.
    """
    preprocessed = []
    for item in chunked:
        if isinstance(item, Tree):
            # Create a new tree, but filter out 'vendor' from the leaves
            filtered_leaves = [(token, pos) for token, pos in item.leaves() if token.lower() != 'vendor']
            if filtered_leaves:
                # Only add the tree if it still contains leaves after filtering
                new_tree = Tree(item.label(), filtered_leaves)
                preprocessed.append(new_tree)
        elif isinstance(item, tuple) and item[0].lower() != 'vendor':
            # Add tuples only if they are not 'vendor'
            preprocessed.append(item)
    return preprocessed

def get_continuous_chunks(text,  tagger, chunk_func=ne_chunk):
    """
    Extracts continuous chunks of named entities or noun phrases from a given text using a specified chunking function.

    This function tokenizes and tags a sentence, applies a chunking function (e.g., NER or NP chunking), and processes 
    the resulting chunks into a list of continuous chunks (noun phrases, named entities). It handles parenthetical content 
    and ensures unique chunk extraction.

    Parameters:
        text (str): The input text to be processed.
        tagger (Flair model): The Part of Speech (POS) tagger.
        chunk_func (function): The chunking function to use for extracting chunks (default is NER chunking).

    Returns:
        list: A list of continuous chunks (noun phrases or named entities) extracted from the text.
    """
    sentence = Sentence(word_tokenize(text))
    # predict NER tags
    tagger.predict(sentence)
    # Extract POS tag tuples from Flair output
    tagged = [(token.text, token.get_labels()[0].value) for token in sentence]
    
    chunked = chunk_func(tagged)
        
    # chunked = preprocess_chunks(chunked)
    continuous_chunk = []
    current_chunk = []

    i = 0
    while i < len(chunked):
        if isinstance(chunked[i], Tree) and chunked[i].label() == 'NP':
            current_chunk.extend(chunked[i].leaves())

        if (isinstance(chunked[i], Tree) and chunked[i].label() == 'NP' and
            i + 1 < len(chunked) and isinstance(chunked[i+1], tuple) and chunked[i+1][0] == '(' and
            i + 2 < len(chunked) and isinstance(chunked[i+2], Tree) and chunked[i+2].label() == 'NP' and
            i + 3 < len(chunked) and isinstance(chunked[i+3], tuple) and chunked[i+3][0] == ')'):
            current_chunk.append('(')
            current_chunk.extend(chunked[i+2].leaves())
            current_chunk.append(')')
            i += 3  # Move index to skip over the parenthetical components

        # Check if it's time to output a chunk and reset for the next one
        if current_chunk and (i + 1 >= len(chunked) or not isinstance(chunked[i+1], Tree)):
            named_entity = " ".join(token if isinstance(token, str) else token[0] for token in current_chunk)
            if named_entity and named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
            current_chunk = []  # Reset for next use

        i += 1  # Natural increment to move to the next item

    return continuous_chunk

def get_sentences(text, nlp):
    """
    Splits a given text into sentences using a natural language processing (NLP) model.

    This function tokenizes the input text into sentences and further splits sentences into smaller parts
    if they contain multiple newlines. It removes empty sentences or whitespace-only parts.

    Parameters:
        text (str): The input text to be split into sentences.
        nlp (Spacy model): The NLP model used for sentence tokenization.

    Returns:
        list: A list of cleaned sentences extracted from the input text.
    """
    # Check for empty or whitespace-only strings
    if not text.strip():
        return []  # Return an empty list if text is empty or contains only whitespace
    
    # Define pattern to split sentences into parts using new lines
    pattern = r'\n\s*\n*'  # Matches one or more new lines with optional whitespace
    doc = nlp(text)
    # Split text into sentences
    sentences = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text:
            # Split each sentence into parts using the defined pattern
            parts = re.split(pattern, sent_text)
            # Add non-empty parts to the list of sentences
            sentences.extend(part.strip() for part in parts if part)
    
    return sentences

def get_contents_noun_phrases(contents, tagger, chunker, nlp):
    """
    Extracts noun phrases from a list of content items by processing each item's sentences.

    This function tokenizes and processes each content item, extracting noun phrases using 
    a chunker and NER tagger. It returns all noun phrases found in the content.

    Parameters:
        contents (list of tuples): A list of content items (tag, text).
        tagger (Flair model): The Part of Speech (POS) tagger.
        chunker (ChunkParser): A chunker used to parse the tokenized sentences.
        nlp (Spacy model): The NLP model used for sentence tokenization.

    Returns:
        list: A list of noun phrases extracted from the content items.
    """
    noun_phrases = []
    for item in contents:
        item_noun_phrases = [get_continuous_chunks(sentence, tagger, chunker.parse) for sentence in get_sentences(item[1], nlp)]
        noun_phrases.append(list(chain.from_iterable(item_noun_phrases)))
    return(noun_phrases)


def get_paragraph_noun_phrases(text, tagger, chunker, nlp):
    """
    Extracts noun phrases from a paragraph of text.

    This function processes a given text paragraph, extracting noun phrases using 
    a chunker and POS tagger. It returns all noun phrases found in the paragraph.

    Parameters:
        text (str): The paragraph of text to be processed.
        tagger (Flair model): The POS tagger.
        chunker (ChunkParser): A chunker used to parse the tokenized sentences.
        nlp (Spacy model): The NLP model used for sentence tokenization.

    Returns:
        list: A list of noun phrases extracted from the paragraph.
    """
    if text=='':
        return []
    noun_phrases = [get_continuous_chunks(sentence, tagger, chunker.parse) for sentence in get_sentences(text)]
    return noun_phrases

def get_sentence_noun_phrases(text, tagger, chunker):
    """
    Extracts noun phrases from a single sentence of text.

    This function processes a given sentence of text, extracting noun phrases using 
    a chunker and NER tagger. It returns all noun phrases found in the sentence.

    Parameters:
        text (str): The sentence to be processed.
        tagger (Flair model): The named entity recognition (NER) tagger.
        chunker (ChunkParser): A chunker used to parse the tokenized sentence.

    Returns:
        list: A list of noun phrases extracted from the sentence.
    """
    if text=='':
        return []
    noun_phrases = get_continuous_chunks(text, tagger, chunker.parse)
    return noun_phrases

# Explode 'Noun Phrases' while preserving all original columns and linking each phrase to its specific content
def explode_noun_phrases(row):
    """
    Explodes a row of noun phrases into multiple rows, preserving original columns.

    This function takes a row containing a list of noun phrases and creates individual rows for each noun phrase,
    while maintaining the original columns (e.g., URL, Header). It allows for easier analysis of each individual 
    noun phrase.

    Parameters:
        row (pandas.Series): A row of data containing noun phrases and associated columns.

    Returns:
        pandas.DataFrame: A DataFrame with exploded rows, where each row corresponds to a single noun phrase.
    """
    # Correctly nested list comprehension for exploding phrases
    exploded_rows = [
        (row['Alternative'], row['URL'], row['Clean_URL'], row['Header'], 
        row['Contents'][idx], idx, phrase, phrase.lower())
        for idx, phrases in enumerate(row['Noun Phrases'])
        for phrase in phrases
    ]
    return pd.DataFrame(exploded_rows, columns=['Alternative', 'URL', 'Clean_URL', 'Header', 'Specific Content', 'Content Index', 'Original Phrase', 'Lower Phrase'])

# Define a function to check for both regular and self-closing HTML tags
def contains_html_tags(content_item):
    """
    Checks whether a content item contains HTML tags (both regular and self-closing).

    This function uses a regular expression to detect both regular and self-closing HTML tags
    in the content. It returns True if any HTML tags are found, and False otherwise.

    Parameters:
        content_item (tuple): A tuple where the second item is the content text to check for HTML tags.

    Returns:
        bool: True if HTML tags are found in the content, False otherwise.
    """
    content = content_item[1]
    # Regex to find regular <tag>...</tag> and self-closing <tag .../> patterns
    pattern = re.compile(r'<(\w+)([^>]*?)\/?>.*?</\1>|<(\w+)([^>]*?)\/>')
    return bool(pattern.search(content))

def process_noun_phrase(row, vendor_name_dict):
    """
    Processes a noun phrase by removing vendor names from it and cleaning the remaining text.

    This function replaces occurrences of vendor names in a given noun phrase with empty strings,
    removes non-alphanumeric characters, and returns the cleaned noun phrase.

    Parameters:
        row (pandas.Series): A row of data containing the original noun phrase.
        vendor_name_dict (dict): A dictionary mapping vendor names to their alternatives.

    Returns:
        str or None: The cleaned noun phrase, or None if the phrase is empty after cleaning.
    """
    # Get the list of vendor names from the dictionary
    vendor_names = vendor_name_dict[row['Alternative']]
    
    # Original phrase
    phrase = row['Original Phrase']
    
    # Iterate through the list of vendor names and replace each one in the phrase
    for vendor_name in vendor_names:
        # Construct a regular expression pattern to match the word/phrase, ignoring case
        pattern = re.compile(re.escape(vendor_name), re.IGNORECASE)
        
        # Replace the vendor name with an empty string
        phrase = re.sub(pattern, '', phrase)
    
    # Clean the text by removing non-alphanumeric characters except some allowed punctuation
    cleaned_text = re.sub(r'[^\w\s,/.:\-\[\]()?&]', '', phrase)
    
    # Remove extra whitespace
    cleaned_text = cleaned_text.strip()
    
    # Check if any characters remain, return None if the phrase is empty
    if not cleaned_text or len(set(cleaned_text)) == 1:
        return None
    
    return cleaned_text



def remove_parentheses(text):
    """
    Removes text enclosed in parentheses from the input string.

    This function uses a regular expression to identify and remove all content inside parentheses, 
    along with the parentheses themselves.

    Parameters:
        text (str): The input string containing parentheses.

    Returns:
        str: The input string with parentheses and their content removed.
    """
    # Use regular expression to find and remove text inside parentheses
    return re.sub(r'\s*\(.*?\)\s*', ' ', text).strip()

def extract_relevant_sentence(content, phrase, nlp):
    """
    Extracts the sentence from a content item that contains the specified phrase.

    This function searches through the sentences of a content item and returns the first sentence
    that contains the given phrase, ignoring any text within parentheses.

    Parameters:
        content (tuple): A tuple where the second item is the content text to be searched.
        phrase (str): The phrase to search for within the sentences.
        nlp (Spacy model): The NLP model used for sentence tokenization.

    Returns:
        str: The sentence containing the phrase, or an empty string if no match is found.
    """
    phrase = remove_parentheses(phrase)
    # sents = get_sentences(content[1])
    doc = nlp(content[1])
    for sent in doc.sents:
        sent_ = remove_parentheses(sent.text)
        if phrase in sent_:
            return sent.text
    return ""  # Return an empty string if no sentence contains the phrase

