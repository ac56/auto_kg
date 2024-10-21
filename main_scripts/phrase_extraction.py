import pandas as pd
import json
import spacy
from tqdm import tqdm
import logging
# import swifter
from flair.models import SequenceTagger
from nltk import RegexpParser
from function_files.phrase_functions import get_contents_noun_phrases, explode_noun_phrases, contains_html_tags, process_noun_phrase, extract_relevant_sentence

def main():
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    vendor_name_dict = config['vendor_name_dict']

    tqdm.pandas()
    logging.getLogger("flair").setLevel(logging.ERROR)
    tagger = SequenceTagger.load("flair/pos-english-fast")
    nlp = spacy.load("en_core_web_sm")

    # Defining a grammar & Parser
    grammar = r"""
        NP: {<NNP|NN>+<CD>}                    # Proper noun followed by a cardinal number
            {<NN|NNP>+<IN><DT>?<NN|NNP>}        # Noun phrase with specific structure, allowing for optional determiner
            {<JJ><NN|NNP|NNS|NNPS>+<CC><NN|NNP|NNS|NNPS>+}
            {<NN|NNP|NNS|NNPS>+(<CC><NN|NNP|NNS|NNPS>+)+}
            {<JJ.*>*<NN.*>+}                   # Adjective(s) (optional) followed by noun(s)
            {<NNP>+<NN>}                       # Proper noun followed by noun
            {<NN.*>+}                          # Any sequence of nouns
            {<NNP|NNPS>+}                      # Proper nouns
            {<NN|NNP|NNS|NNPS>+}               # Noun or proper noun sequences
    """

    chunker = RegexpParser(grammar)

    contents_df = pd.read_pickle('web_data/dense_contents')
    contents_df['Noun Phrases'] = contents_df['Contents'].progress_apply(lambda x: get_contents_noun_phrases(x, tagger, chunker, nlp))

    noun_phrases_list = []
    # Iterate over the DataFrame rows and track progress
    for content in tqdm(contents_df['Contents'], desc="Processing noun phrases"):
        noun_phrases = get_contents_noun_phrases(content, tagger, chunker, nlp)
        noun_phrases_list.append(noun_phrases)

    # Add the results to the DataFrame
    contents_df['Noun Phrases'] = noun_phrases_list

    # Apply the function and concatenate results
    df_exploded = pd.concat([explode_noun_phrases(row) for index, row in contents_df.iterrows()], ignore_index=True)

    # Apply the function to the 'Specific Content' column and create a mask
    mask = df_exploded['Specific Content'].apply(contains_html_tags)

    # Filter the DataFrame based on the mask
    df_exploded = df_exploded[~mask]

    df_exploded.drop_duplicates(subset=['Alternative', 'Lower Phrase'], keep='first', inplace=True)
    df_exploded.reset_index(drop=True, inplace=True)

    df_exploded['Processed Phrase'] = df_exploded.apply(lambda row: process_noun_phrase(row, vendor_name_dict), axis=1)

    df_exploded = df_exploded.dropna(subset=['Processed Phrase'])
    df_exploded.reset_index(drop=True, inplace=True)

    # Apply the function to create 'Context' column
    df_exploded['Context'] = df_exploded.apply(lambda row: extract_relevant_sentence(row['Specific Content'], row['Original Phrase'], nlp), axis=1)

    df_exploded.to_pickle('web_data/phrase_data_')
    df_exploded.to_csv('web_data/phrase_data_.csv', index=False)

if __name__ == '__main__':
    main()