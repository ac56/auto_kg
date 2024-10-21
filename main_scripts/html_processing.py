import json
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from function_files.content_functions import organize_headers_content, column_agg


def main():
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    repeat_limit = config['repeat_limit']

    data_df = pd.read_pickle('web_data/web_content')
    result_df = pd.DataFrame(columns=['Alternative', 'URL', 'Clean_URL', 'Previous_Header', 'Header','Contents'])

    # Prepare the list to collect new rows for the resulting DataFrame
    new_rows = []

    # Iterate through each row in the merged DataFrame
    for idx, row in data_df.iterrows():
        html_content = row['HTML_Content']
        soup = BeautifulSoup(html_content, 'lxml')
        try:
            header_content_pairs = organize_headers_content(soup)
        except Exception as e:
            print(f"Error processing HTML content: {e}")
            header_content_pairs = []  # Ensure it's empty to skip content-related processing

        # Track the previous header
        previous_header = None

        if not header_content_pairs:  # If empty due to an error or no headers found
            new_row = {
                'Alternative': row['Alternative'],
                'URL': row['URL'],
                'Clean_URL': row['Clean_URL'],
                'Previous_Header': None,
                'Header': None,
                'Contents': None
            }
            new_rows.append(new_row)
            continue  # Skip to the next iteration

        for header, contents in header_content_pairs:
            if contents:  # Only consider headers with contents
                new_row = {
                    'Alternative': row['Alternative'],
                    'URL': row['URL'],
                    'Clean_URL': row['Clean_URL'],
                    'Previous_Header': previous_header,
                    'Header': header,
                    'Contents': contents
                }
                new_rows.append(new_row)
            
            # Update previous header for the next iteration
            previous_header = header if not contents else previous_header

    # Create or append to a new DataFrame from the list of new rows
    if new_rows:
        new_result_df = pd.DataFrame(new_rows)
        result_df = pd.concat([result_df, new_result_df])

    # result_df.to_pickle('web_data/result_df')
    # result_df.to_csv('web_data/result_df.csv', index=False)
    
    # result_df = pd.read_pickle('web_data/result_df')
    data_df = result_df.copy()

    # Filter rows where 'Contents' is either None or an empty list
    empty_contents_df = data_df[data_df['Contents'].apply(lambda x: x == [] or x is None)]

    # Get the indices of rows with empty or None values in the 'Contents' column
    indices_to_drop = empty_contents_df.index

    # Drop the rows from the original DataFrame and reset the index
    filtered_data_df = data_df.drop(indices_to_drop).reset_index(drop=True)

    # Safely convert 'Contents' to tuples, handling None values
    filtered_data_df['Contents_Tuple'] = filtered_data_df['Contents'].apply(lambda x: tuple(x) if x is not None else ())

    # Now proceed to drop duplicates
    filtered_data_df = filtered_data_df.drop_duplicates(subset=['Clean_URL', 'Header', 'Contents_Tuple'])

    filtered_data_df.reset_index(inplace=True, drop=True)
    # Sort the DataFrame by 'Contents_Tuple' in descending order
    filtered_data_df.sort_values(by=['Contents_Tuple'], ascending=False, inplace=True)

    # Now proceed to drop duplicates based on 'Clean_URL' and 'Header'
    filtered_data_df.drop_duplicates(subset=['Clean_URL', 'Header'], keep='first', inplace=True)

    # Reset the index after dropping duplicates
    filtered_data_df.reset_index(drop=True, inplace=True)

    # Define the aggregation dictionary for each column
    aggregation_dict = {
        'Alternative': column_agg,
        'URL': lambda x: list(x),
        'Clean_URL': lambda x: list(x),
        'Previous_Header': lambda x: None,
    }

    # Group by 'Header' and 'Contents_Tuple' and aggregate the DataFrame
    aggregated_df = filtered_data_df.groupby(['Header', 'Contents_Tuple']).agg(aggregation_dict).reset_index()

    # Create a dictionary mapping from unique combinations of 'Header' and 'Contents_Tuple' to their corresponding 'Contents' values
    contents_mapping = filtered_data_df.groupby(['Header', 'Contents_Tuple'])['Contents'].first().to_dict()

    # Map the 'Contents' values from the dictionary to the corresponding rows in the aggregated DataFrame
    aggregated_df['Contents'] = aggregated_df.apply(lambda row: contents_mapping.get((row['Header'], row['Contents_Tuple']), None), axis=1)

    # Get the column order from filtered_data_df
    column_order = filtered_data_df.columns.tolist()

    # Reorder the columns in the aggregated DataFrame
    aggregated_df = aggregated_df[column_order]
    aggregated_df.drop(columns=['Previous_Header'], inplace=True)

    # Initialize a Counter to count occurrences of content items across all rows
    content_counter = Counter()

    # Initialize a defaultdict to store the 'Clean_URL' and 'Header' for each repeated content item
    repeated_content_info = defaultdict(lambda: {'Count': 0, 'Clean_URL': [], 'Header': [], 'Content': None})

    # Iterate over each row in the DataFrame
    for index, row in aggregated_df.iterrows():
        # Skip rows where 'Contents' is None or empty
        if not row['Contents']:
            continue
        
        # Increment the count of each content item across all rows
        content_counter.update(row['Contents'])

    # Iterate over the content items and their counts
    for content, count in content_counter.items():
        # Check if the content item is repeated
        if count > 1:
            # Increment the count of repetitions for the content item
            repeated_content_info[content]['Count'] += count
            # Append the 'Clean_URL' and 'Header' for the repeated content item
            repeated_content_info[content]['Clean_URL'].extend(aggregated_df[aggregated_df['Contents'].apply(lambda x: content in x)]['Clean_URL'])
            repeated_content_info[content]['Header'].extend(aggregated_df[aggregated_df['Contents'].apply(lambda x: content in x)]['Header'])
            # Save the content itself
            repeated_content_info[content]['Content'] = content

    # Create a DataFrame from the repeated content information
    repeated_content_df = pd.DataFrame(repeated_content_info).T
    morethan = repeated_content_df[repeated_content_df['Count'] > repeat_limit]

    # Iterate over each row in the DataFrame
    for index, row in aggregated_df.iterrows():
        # Skip rows where 'Contents' is None or empty
        if not row['Contents']:
            continue
        
        # Filter out content items with count greater than 10
        row['Contents'] = [content for content in row['Contents'] if content not in morethan.index]

    # Filter rows where 'Contents' is either None or an empty list
    empty_agg_df = aggregated_df[aggregated_df['Contents'].apply(lambda x: x == [] or x is None)]

    # Get the indices of rows with empty or None values in the 'Contents' column
    indices_to_drop = empty_agg_df.index

    # Drop the rows from the original DataFrame and reset the index
    aggregated_df = aggregated_df.drop(indices_to_drop).reset_index(drop=True)

    # Drop the 'Contents_Tuple' as it is no longer needed
    aggregated_df.drop('Contents_Tuple', axis=1, inplace=True)

    aggregated_df['Clean_URL_Tuple'] = aggregated_df['Clean_URL'].apply(tuple)

    # Sort the DataFrame based on the 'Alternative' and 'Clean_URL_Tuple' columns
    aggregated_df.sort_values(by=['Alternative', 'Clean_URL_Tuple'], inplace=True)

    # Drop the temporary column
    aggregated_df.drop(columns=['Clean_URL_Tuple'], inplace=True)
    aggregated_df.reset_index(inplace=True, drop=True)

    aggregated_df.to_pickle('web_data/dense_contents')
    aggregated_df.to_csv('web_data/dense_contents.csv', index=False)


if __name__ == '__main__':
    main()