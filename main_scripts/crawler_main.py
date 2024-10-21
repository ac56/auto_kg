import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# import os
from urllib.parse import urljoin, urlunsplit, urlsplit
import requests.exceptions

from function_files.crawler_functions import is_relative_url, normalize_url, is_relevant_link, check_url_relevancy, filter_urls_by_extension, get_html_content, can_scrape, clean_url

def main():
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    include_terms = config['include_terms']
    exclude_terms = config['exclude_terms']
    excluded_extensions = config['excluded_extensions']

    df_urls = pd.read_csv("init_data/vendor_urls.csv")
 

    extracted_urls = []
    # Loop through each URL and extract relevant links
    for i in tqdm(range(len(df_urls))):
        alternative = df_urls.loc[i, 'Alternative']
        base_url = df_urls.loc[i, 'URL']
        # print(base_url) 
        
        parsed_url = urlsplit(base_url)
        base_path = urlunsplit((parsed_url.scheme, parsed_url.netloc, '', '', ''))

        # Initialize a list of URLs to visit
        to_visit = [base_url]
        
        # Check if scraping is allowed
        is_allowed_to_scrape = can_scrape(base_url)
        
        if is_allowed_to_scrape:
            try:
                reqs = requests.get(base_url)
                soup = BeautifulSoup(reqs.text, 'html.parser')
            except Exception as e:
                print(f"Error processing {base_url}: {e}")
                continue  # Skip to next URL if there's an error
            
            urls = []
            
            # Extract all links from the page
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None:
                    href = normalize_url(href)
                    if is_relative_url(href):
                        href = urljoin(base_path, href)
                    urls.append(href) 
            
            # Filter the links based on relevance and uniqueness
            filtered_links = [link for link in urls if is_relevant_link(link, base_url)]
            new_urls = [link for link in set(filtered_links) if link not in to_visit]
            to_visit.extend(new_urls)
            
            # Filter out URLs based on excluded extensions
            filtered_urls = filter_urls_by_extension(to_visit, excluded_extensions)
            
            # First row: store the original URL
            extracted_urls.append({'Alternative': alternative, 'URL': base_url})
            
            # Filter relevant links
            relevant_links = [link for link in filtered_urls if check_url_relevancy(link, include_terms, exclude_terms)]
            filtered_urls = filter_urls_by_extension(relevant_links, excluded_extensions)
            
            # Add all relevant links to the dataframe for the same Alternative
            for link in filtered_urls:
                extracted_urls.append({'Alternative': alternative, 'URL': link})
        else:
            print(f'Access to URL: {base_url} was denied.')

    # Convert the final data into a DataFrame
    url_df = pd.DataFrame(extracted_urls, columns=['Alternative', 'URL'])
    url_df['Clean_URL'] = url_df['URL'].apply(clean_url)
    url_df.drop_duplicates(subset=['Clean_URL'], keep='first', inplace=True)

    url_df.to_csv('web_data/url_df.csv', index=False)
    url_df.to_pickle('web_data/url_df')
        
    # url_df = pd.read_pickle('web_data/url_df')
    

    htmls = []
    # Get web content
    for idx, row in url_df.iterrows():
        url = row['URL']
        html_content = get_html_content(url)
        if html_content:
            htmls.append(html_content)
        else:
            htmls.append('Failed to retrieve HTML content')

    url_df['HTML_Content'] = htmls
    url_df.to_pickle('web_data/web_content')
    url_df.to_csv('web_data/web_content.csv', index=False)

if __name__ == '__main__':
    main()