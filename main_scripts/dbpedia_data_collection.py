import pandas as pd
from function_files.dbpedia_functions import run_sparql_query, construct_query_for_subject_of, construct_query_for_linked_from, process_results, get_concept_name, get_concept_list, get_resource_list, create_resource_dataframe, get_resource_names, explode_and_label, correct_name

def main():
    concepts_df = pd.read_csv('init_data/concepts.csv')
    all_concepts = pd.concat([concepts_df['Concept'], concepts_df['NarrowerConcept']]).unique()

    concept_info = pd.DataFrame(columns=['Concept', 'SubjectOf', 'LinkedFrom'])
    for concept_uri in all_concepts:
        # Make sure concept_uri is a valid URI and not NaN or None
        if pd.notna(concept_uri):
            # Your SPARQL query execution and result processing logic here
            subject_of_results = run_sparql_query(construct_query_for_subject_of(concept_uri))
            linked_from_results = run_sparql_query(construct_query_for_linked_from(concept_uri))
            
            new_row = pd.DataFrame({'Concept': [concept_uri], 'SubjectOf': [process_results(subject_of_results)], 'LinkedFrom': [process_results(linked_from_results)]})
            concept_info = pd.concat([concept_info, new_row], ignore_index=True)

    # Apply the function to each URI in the lists of the 'LinkedResources' column
    concept_info['Name'] = concept_info['Concept'].apply(lambda uri: get_concept_name(uri))
    concept_order = ['Concept', 'Name', 'SubjectOf', 'LinkedFrom']
    concept_info = concept_info[concept_order]

    concept_info.to_pickle('dbpedia/concept_info')
    concept_info.to_csv('dbpedia/concept_info.csv', index=False)
    
    # concept_info = pd.read_pickle('dbpedia/concept_info')

    concept_list = get_concept_list(concepts_df)
    resource_list = get_resource_list(concept_info)
    all_resources = list(set().union(*concept_info['SubjectOf'], *concept_info['LinkedFrom']))

    resource_info, failed_to_parse, failed_to_query = create_resource_dataframe(all_resources)
    resource_info = get_resource_names(resource_info)

    resource_info.to_csv('dbpedia/resource_info.csv', index=False)
    resource_info.to_pickle('dbpedia/resource_info')

    # resource_info = pd.read_pickle('dbpedia/resource_info')

    # Exploding list-type mappings
    genres = explode_and_label(resource_info, 'Genre(s)', 'GenreNames', 'resource')
    genres_of = explode_and_label(resource_info, 'GenreOf', 'GenreOfNames', 'resource')
    linked = explode_and_label(resource_info, 'LinkedResources', 'LinkedNames', 'resource')
    disambiguate_of = explode_and_label(resource_info, 'DisambiguateOf', 'DisambiguateOfNames', 'resource')
    product = explode_and_label(resource_info, 'Product(s)', 'ProductNames', 'resource')
    product_of = explode_and_label(resource_info, 'ProductOf', 'ProductOfNames', 'resource')

    simple_resource = pd.DataFrame({
        'URI': resource_info['Resource'],
        'Name': resource_info['Label'],
        'Type': 'resource'
    })

    simple_concept = pd.DataFrame({
        'URI': concept_info['Concept'],
        'Name': concept_info['Name'],
        'Type': 'concept'
    })

    # Concatenate all dataframes
    dbpedia_vocab = pd.concat([simple_resource, simple_concept, genres, genres_of, linked, disambiguate_of, product, product_of])

    # Drop duplicates based on URI and Name columns
    dbpedia_vocab = dbpedia_vocab.drop_duplicates(subset=['URI', 'Name'])
    dbpedia_vocab.reset_index(drop=True, inplace=True)
    dbpedia_vocab['Name'] = dbpedia_vocab['Name'].apply(lambda name: correct_name(name))

    dbpedia_vocab.to_csv('dbpedia/dbpedia_vocab.csv', index=False)
    dbpedia_vocab.to_pickle('dbpedia/dbpedia_vocab')

if __name__ == '__main__':
    main()