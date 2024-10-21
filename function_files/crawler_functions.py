import json
import re
from selenium import webdriver
from selenium.common import exceptions
from urllib import robotparser
from urllib.parse import urljoin, urlunsplit, urlsplit

def configSelenium():
    """
    Configures Selenium WebDriver with necessary options for headless browsing.

    Returns:
        webdriver.Chrome: A Selenium Chrome WebDriver configured for headless browsing.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-infobars')
    options.add_argument('--no-sandbox')

    driver = webdriver.Chrome(options=options)
    return driver

def can_scrape(start_url):
    """
    Checks if a given URL can be scraped by verifying the robots.txt rules.

    Parameters:
        start_url (str): The URL to check scraping permissions for.

    Returns:
        bool: True if scraping is allowed, False otherwise.
    """

    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(start_url, '/robots.txt'))
    rp.read()
    user_agent = '*'
    return rp.can_fetch(user_agent, start_url)

def is_relative_url(href):
    """
    Checks if a given URL is a relative URL.

    Parameters:
        href (str): The URL to check.

    Returns:
        bool: True if the URL is relative, False otherwise.
    """
    return not (href.startswith('http://') or href.startswith('https://') or href.startswith('http:/') or href.startswith('https:/'))

def normalize_url(url):
    """
    Normalizes a URL by removing query parameters and fragments.

    Parameters:
        url (str): The URL to normalize.

    Returns:
        str: The normalized URL.
    """
    url = url.lstrip()
    url = url.rstrip()
    parts = urlsplit(url)
    # Rebuild URL without query and fragment
    return urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))

def clean_url(url):
    """
    Cleans a URL by removing the protocol, 'www', and any trailing slashes.

    Parameters:
        url (str): The URL to clean.

    Returns:
        str: The cleaned URL.
    """
    # Remove the protocol and www
    cleaned_url = re.sub(r'https?://(www\.)?', '', url)
    # Remove trailing slashes
    cleaned_url = re.sub(r'/$', '', cleaned_url)
    return cleaned_url

def extract_path_criteria(url):
    """
    Extracts the first significant part of a URL's path.

    Parameters:
        url (str): The URL to extract the path from.

    Returns:
        str: The first significant part of the URL's path.
    """
    parsed_url = urlsplit(url)
    path_parts = parsed_url.path.split('/')
    # Handle URLs without a significant path component
    if len(path_parts) <= 2 or path_parts[1] == '':
        return '/'
    # Extract the first significant part of the path
    return '/' + path_parts[1] + '/' 

def is_relevant_link(url, base_url):
    """
    Determines if a link is relevant by comparing its domain and path to a base URL.

    Parameters:
        url (str): The URL to check.
        base_url (str): The base URL to compare against.

    Returns:
        bool: True if the link is relevant, False otherwise.
    """
    parsed_url = urlsplit(url)
    base_domain = "{0.netloc}".format(urlsplit(base_url))
    path_criteria = extract_path_criteria(base_url)

    return parsed_url.netloc == base_domain and (path_criteria in parsed_url.path or path_criteria == '/') 


def extract_url_parts(input_url):
    """
    Extracts meaningful parts of a URL by splitting its path and excluding unnecessary parts.

    Parameters:
        input_url (str): The input URL to extract parts from.

    Returns:
        list: A list of relevant parts extracted from the URL.
    """
    parsed_url = urlsplit(input_url)
    domain = '.'.join(parsed_url.netloc.split('.')[1:2])
    path_parts = [part for part in parsed_url.path.split('/') if part]
    split_path_parts = []
    for part in path_parts:
        words = part.split('-')  # Split on hyphens to separate words
        if len(words) > 1:  # If there are multiple words, create a tuple
            split_path_parts.extend(words)
        else:
            split_path_parts.append(part)
    split_path_parts = [part for part in split_path_parts if len(part)>2]
    split_path_parts = [part for part in split_path_parts if domain not in part]
    return split_path_parts

def check_url_relevancy(input_url, check_list, exclude_terms):
    """
    Checks if a given URL is relevant based on a list of criteria and exclusion terms.

    Parameters:
        input_url (str): The URL to check.
        check_list (list of str): The list of criteria to check against.
        exclude_terms (list of str): The list of terms to exclude.

    Returns:
        bool: True if the URL is relevant, False otherwise.
    """
    path_parts = extract_url_parts(input_url)
    for part in path_parts:
        for exc_term in exclude_terms:
            if exc_term in part:
                return False
    for part in path_parts:
        for test in check_list:
            test_parts = test.split()
            for test_part in test_parts:
                if(test_part in part):
                    return True
    return False

def remove_punctuation_regex(text):
    """
    Removes punctuation from a string using regular expressions.

    Parameters:
        text (str): The input text to clean.

    Returns:
        str: The text with punctuation removed.
    """
    # Use regular expression to remove punctuation
    return re.sub(r'[^\w\s-]', '', text)


def filter_urls_by_extension(url_list, excluded_extensions):
    """
    Filters out URLs with certain extensions from a list of URLs.

    Parameters:
        url_list (list of str): The list of URLs to filter.
        excluded_extensions (list of str): The list of extensions to exclude.

    Returns:
        list: The filtered list of URLs.
    """
    filtered_urls = [url for url in url_list if not any(url.endswith(ext) for ext in excluded_extensions)]
    return filtered_urls

def saveJsonObject(jsonObject,filename):
    """
    Saves a JSON object to a file.

    Parameters:
        jsonObject (dict): The JSON object to save.
        filename (str): The name of the file to save the JSON object to.

    Returns:
        None
    """
    f = open("./"+filename, "w+")
    f.write(json.dumps(jsonObject, indent=2))
    f.close()

def get_html_content(url):
    """
    Fetches the HTML content of a webpage using Selenium WebDriver.

    Parameters:
        url (str): The URL of the webpage to fetch.

    Returns:
        str: The HTML content of the page, or None if an error occurs.
    """
    driver = configSelenium()
    try:
        driver.get(url)
        driver.implicitly_wait(0.5)
        html_content = driver.page_source
        return html_content
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None
    finally:
        driver.quit()

def drop_url_duplicates(df, column):
    """
    Removes duplicate rows in a DataFrame based on a given column and retains the row with the shortest URL.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing URLs and content.
        column (str): The column to check for duplicate content.

    Returns:
        pandas.DataFrame: The DataFrame with duplicates removed.
    """
    # Assuming web_content is your DataFrame containing URLs and HTML content
    duplicate_html_content = df.groupby(column)['URL'].filter(lambda x: len(x) > 1)

    # Group the URLs by their HTML content
    grouped_urls = df.loc[df['URL'].isin(duplicate_html_content)].groupby(column)['URL']

    # Find indices where HTML content is duplicated
    duplicate_html_content_indices = df[df.duplicated(column, keep=False)].index.tolist()
    # Create a list to collect indices of the shortest URLs in each group
    indices_to_keep = []

    for html_content, urls in grouped_urls:
        # Find the index of the row with the shortest URL in the group
        shortest_index = urls.str.len().idxmin()
        indices_to_keep.append(shortest_index)

    # Convert indices of all duplicates and indices to keep into sets
    all_duplicate_indices = set(duplicate_html_content_indices)
    indices_to_keep_set = set(indices_to_keep)

    # Calculate the indices to drop (all duplicates minus the ones we want to keep)
    indices_to_drop = all_duplicate_indices - indices_to_keep_set

    # Drop the rows that are in indices_to_drop from the original DataFrame
    df = df.drop(index=indices_to_drop)

    return df