import re
from bs4 import BeautifulSoup, NavigableString, Tag



def filter_cookie_content(header_content_pairs):
    """
    Filters out content related to cookies or consent from the given header-content pairs.

    This function processes a list of header-content pairs, where each pair consists of a header
    and a list of content items. It removes any content that contains references to "cookie" or "consent"
    (case-insensitive) in either the header or the content items themselves.

    Parameters:
        header_content_pairs (list of tuples): A list where each tuple contains a header (string) 
        and a list of content items (each content item is a tuple).

    Returns:
        list of tuples: A list of filtered header-content pairs, where content related to cookies or consent 
        has been removed.
    """
    filtered_pairs = []
    pattern = re.compile(r'cookie|consent', re.IGNORECASE)
        
    for header, content in header_content_pairs:
        if not (pattern.search(header)):
            filtered_content = [item for item in content if not pattern.search(item[1])]
            filtered_pairs.append((header, filtered_content))
    return filtered_pairs

def filter_content(header_content_pairs):
    """
    Filters out content items with unidentified htmk tag ('u') from the given header-content pairs.

    Parameters:
        header_content_pairs (list of tuples): A list where each tuple contains a header (string) and a 
        list of content items (each content item is a tuple).

    Returns:
        list of tuples: A list of header-content pairs
    """
    for i, pair in enumerate(header_content_pairs):
        content = pair[1]
        if any(c[0] != 'u' for c in content):
            new_content = [c for c in content if c[0] != 'u']
            header_content_pairs[i] = (pair[0], new_content)
    return header_content_pairs  

def handle_overlaps(header_content_pairs):
    """
    Handles overlapping content between adjacent header-content pairs.

    This function checks for overlaps in content between adjacent header-content pairs. It removes any 
    content that is repeated across pairs, such as when the last content of a current header contains 
    the first content of the next header. Additionally, it ensures that content from the current header 
    that appears in the next header is not duplicated.

    Parameters:
        header_content_pairs (list of tuples): A list where each tuple contains a header (string) and a 
        list of content items (each content item is a tuple).

    Returns:
        list of tuples: A modified list of header-content pairs with overlapping or duplicate content removed.
    """
    # Iterate over the header content pairs
    for i in range(len(header_content_pairs) - 1):  # Exclude the last header as it has no next header
        current_header, current_contents = header_content_pairs[i]
        next_header, next_contents = header_content_pairs[i + 1]

        # Ensure there are contents to check
        if current_contents and next_contents:
            last_content = current_contents[-1][1]  # Last content of the current header
            first_next_content = next_contents[0][1]  # First content of the next header

            # Check if the last content of the current header contains the first content of the next header
            if first_next_content in last_content:
                # Remove the overlapping content
                current_contents.pop()

            # Update the modified contents back to the list
            header_content_pairs[i] = (current_header, current_contents)

            # Remove duplicate entries from the current header that appear in the next header
            unique_current_contents = []
            next_contents_texts = [content[1] for content in next_contents]  # Extract just the text for comparison

            for content in current_contents:
                if content[1] not in next_contents_texts:
                    unique_current_contents.append(content)
            # Update the modified contents back to the list
            header_content_pairs[i] = (current_header, unique_current_contents)
    return header_content_pairs


def adjust_spaces(text):
    """ Removes unnecessary spaces from a string """
    space_pattern = r'\u00A0|\u200B|\u2002|\u2003|\u2004|\xa0|\u200b'
    # Normalize other unicode space characters
    text = re.sub(space_pattern, ' ', text)
    # Replace multiple spaces with a single space but preserve new lines
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Preserve new lines if they are not followed by a lowercase letter even with spaces in between
    text = re.sub(r'\n\s*(?=[a-z])', ' ', text)
    # Remove leading and trailing spaces
    text = text.strip()

    return text

def normalize_text(input_string):
    """ Standardizes input strings for comparison purposes """

    space_pattern = r'\u00A0|\u200B|\u2002|\u2003|\u2004|\xa0|\u200b'
    # Remove punctuation using regular expression
    cleaned_string = re.sub(r'[^\w\s]', '', input_string)
    # Remove extra spaces
    cleaned_string = re.sub(space_pattern, ' ', cleaned_string)
    cleaned_string = re.sub(r'\s+', '', cleaned_string)
    # Strip leading and trailing spaces
    cleaned_string = cleaned_string.strip()
    return cleaned_string

def check_membership(element, contents):
    """
    Check if the items in contents collectively cover all the characters in the element.
    Additionally, check if all items in contents start with a capitalized letter.
    Returns a boolean indicating coverage and capitalization conditions, and indices of items that cover the element.
    """
    covered = [False] * len(element)  # Tracking coverage of characters in element
    member_indices = []
    members = []

    # Check character coverage by contents
    for index, item in enumerate(contents):
        if item in element:
            members.append(item)
            start_index = element.find(item)  # Find each item in the element
            if start_index != -1:
                # Mark the span of characters covered by this item
                end_index = start_index + len(item)
                for i in range(start_index, min(end_index, len(element))):  # Ensure not to exceed element length
                    covered[i] = True
                member_indices.append(index)  # Collect indices of contributing items

    # Check if all characters are covered
    all_covered = all(covered)
    
    # Check if all items start with a capital letter or symbol
    no_lowercase = not any(item[0].islower() for item in members if item)

    return (all_covered and no_lowercase), member_indices

def update_contents_for_header(header, elements, header_content_pairs):
    """
    Updates content items associated with a given header, replacing or appending content based on overlap or 
    membership with existing content. The function checks if the content already exists and either updates or 
    appends new elements to avoid redundancy.

    Parameters:
        header (str): The header whose content needs to be updated.
        elements (list of tuples): A list of content items to be added or updated, where each item is a tuple.
        header_content_pairs (list of tuples): A list of existing header-content pairs to update.

    Returns:
        list: The updated content for the given header, after handling overlaps and ensuring uniqueness.
    """
    if len(elements)>20:
        elements = elements[:20]

    content_pair = next((pair for pair in header_content_pairs if pair[0] == header), None)
    if not content_pair:
        # If no existing header matches, possibly initialize or skip updating
        return elements
    contents = content_pair[1]

    for i, element_pair in enumerate(elements):
            normal_e = normalize_text(element_pair[1])
            normal_contents = [normalize_text(content[1]) for content in contents]
            membership, member_indices = check_membership(normal_e, normal_contents)
            
            if not membership:
                # Proceed only if membership test fails and member_indices is not empty
                if member_indices:
                    # Replace the first overlapping content and remove subsequent ones
                    if member_indices[0] < len(contents):
                        contents[member_indices[0]] = element_pair
                    for j in sorted(member_indices[1:], reverse=True):
                        if j < len(contents):
                            del contents[j]
                    # Manage overlap with next element
                    next_index = member_indices[-1] + (2 - len(member_indices))
                    if next_index < len(contents) and (i + 1) < len(elements):
                        next_element_pair = elements[i + 1]
                        next_element_text = next_element_pair[1]
                        next_normal_e = normalize_text(next_element_text)
                        if normal_contents[next_index] in (normal_e + next_normal_e):
                            if next_index < len(contents):
                                contents[next_index] = next_element_pair
                                # Optionally remove the old content that overlaps significantly with the next element
                                if next_index + 1 < len(contents):
                                    del contents[next_index + 1]

                elif not any(normal_e in content for content in normal_contents):
                    contents.append(element_pair) 
             
    return contents

def find_header_for_element(element):
    """
    Finds the closest header tag (e.g., h1, h2, h3, etc.) for a given element by traversing upward through the DOM.

    This function traverses upwards from the given element, collecting all preceding sibling headers, and returns
    the first header encountered with non-empty text.

    Parameters:
        element (Tag): A BeautifulSoup Tag object representing the element for which to find the nearest header.

    Returns:
        str: The text of the nearest header, or None if no header is found.
    """
    current = element
    headers_before_element = []

    # Traverse upwards from the element, collecting all headers
    while current:
        # Collect all headers within the current element and its preceding siblings
        for sibling in current.find_all_previous():
            if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                headers_before_element.append(sibling)

        current = current.parent  # Move up one level in the DOM

    # Check headers collected that are before the element in source order
    for header in headers_before_element:
        header_text = header.text.strip()
        if header_text:  # Return the first non-empty header
            return header_text
    return None  # Return None if all headers are empty or no headers are found

def get_path_to_root(element):
    """
    Constructs a path from a given element to the root of the document (HTML tag).

    This function builds a list of elements by traversing upward from the given element to its root (HTML).
    The list contains each element encountered in this traversal, ending with the root element.

    Parameters:
        element (Tag): A BeautifulSoup Tag object representing the element for which to trace the path to the root.

    Returns:
        list: A list of Tag objects representing the path from the element to the root.
    """
    path = []
    while element:
        path.append(element)
        element = element.parent
    return path

def filter_elements(elements):
    """
    Filters out unwanted elements based on their class, id, or content that matches certain patterns 
    (e.g., 'footer', 'cookies', 'contact', etc.).

    This function checks for unwanted elements, such as footers or cookie notifications, by searching for 
    matching patterns in the text, class, or id attributes. Elements that don't match the pattern are retained.

    Parameters:
        elements (list of Tag objects): A list of BeautifulSoup Tag objects to filter.

    Returns:
        list: A filtered list of elements that don't match the unwanted patterns.
    """
    # Define a regex pattern to match 'footer' or 'cookie' in class or id
    pattern = re.compile(r'\b(footer|cookies|consent|contact|form|sitemap|mobile|menu)\b', re.IGNORECASE) # |news|dropdown|form
    selected = []
    for element in elements:
        if not (pattern.search(element.text.strip()) or (element.get('class') and any(pattern.search(item) for item in element.get('class'))) or \
           (element.get('id') and pattern.search(element.get('id')))):
            path  = get_path_to_root(element)
            found = False
            body_in_path = False
            for part in path:
                if part.name == 'body':
                    body_in_path = True
                if part.name != 'html' and not body_in_path:
                    if (part.get('class') and any(pattern.search(item) for item in part.get('class'))) or \
                (part.get('id') and pattern.search(part.get('id'))):
                        found = True
                        break
            if not found:
                selected.append(element)
    return selected

def group_elements_by_headers(soup, tag):
    """
    Groups elements by their nearest header tag (e.g., h1, h2, etc.).

    This function finds all elements of a specific tag (e.g., div, p) and associates each with its nearest 
    preceding header (h1, h2, etc.). The result is a dictionary where the headers are keys, and the associated 
    elements are values.

    Parameters:
        soup (BeautifulSoup): A BeautifulSoup object representing the parsed HTML document.
        tag (str): The tag to search for and group by headers (e.g., 'div', 'p').

    Returns:
        dict: A dictionary where keys are header texts and values are lists of tuples (tag, content).
    """
    header_element_map = {}
    elements = soup.find_all(tag)
    elements = filter_elements(elements)
    if len(elements)>20:
        elements = elements[:20]

    for element in elements:
        header_text = find_header_for_element(element)
        if header_text:
            if header_text not in header_element_map:
                header_element_map[header_text] = []
            if tag == 'div':
                texts = []
                for content in element.contents:
                    if isinstance(content, NavigableString) and not isinstance(content, Tag):
                        text = content.strip()
                        if text:  # Ensure the string is not just empty or whitespace
                            texts.append(text)
                div_text = ' '.join([string for string in texts])
                if(div_text):
                    header_element_map[header_text].append(('str', div_text))
                else:
                    continue
            else:
                header_element_map[header_text].append((tag, element.text.strip()))

    return header_element_map

def unduplicate_content(header_content_pairs):
    """
    Removes duplicate content items from the header-content pairs.

    This function ensures that content items associated with a given header are unique, removing duplicates 
    that have already been seen in previous headers.

    Parameters:
        header_content_pairs (list of tuples): A list of header-content pairs to process.

    Returns:
        list of tuples: A list of header-content pairs with duplicates removed.
    """
    seen_pairs = {}
    unique_pairs = []
    for header, content in header_content_pairs:
        seen_items = set()
        unique_items = []
        for item in content:
            if item[1] not in seen_items:
                unique_items.append(item)
                seen_items.add(item[1])


        pair_key = (header, tuple(unique_items))  # Convert list to tuple for hashing

        if pair_key not in seen_pairs:
            seen_pairs[pair_key] = True
            unique_pairs.append((header, unique_items))

    return unique_pairs

def content_cap(header_content_pairs, cap_limit):
    """ Truncates the content collected for each header based on a content cap/ limit """
    for i, pair in enumerate(header_content_pairs):
        content = pair[1]
        if len(content)>cap_limit:
            new_content = content[:cap_limit]
            header_content_pairs[i] = (pair[0], new_content)
    return header_content_pairs

def refine_content(soup, header_content_pairs):
    """
    Refines the extracted content by applying several filtering and processing steps.

    This function processes header-content pairs by capping the number of items, removing duplicates, 
    and updating the content using text elements grouped by headers (e.g., 'p', 'li', 'div'). 
    It adjusts spaces, handles overlaps, and filters out unnecessary content.

    Parameters:
        soup (BeautifulSoup): A BeautifulSoup object representing the parsed HTML content.
        header_content_pairs (list of tuples): A list of header-content pairs to be refined.

    Returns:
        list of tuples: A refined list of header-content pairs, with duplicates removed and content adjusted.
    """
    header_content_pairs = content_cap(header_content_pairs, 20)
    header_content_pairs = unduplicate_content(header_content_pairs)

    tags = ['p', 'li', 'div']
    for tag in tags:

        header_element_map = group_elements_by_headers(soup, tag)

        for header, elements in header_element_map.items():
            updated_contents = update_contents_for_header(header, elements, header_content_pairs)

            for i, pair in enumerate(header_content_pairs):
                if pair[0] == header:
                    header_content_pairs[i] = (header, updated_contents)

    for i, pair in enumerate(header_content_pairs):
        adjusted_contents = [(item[0],adjust_spaces(item[1])) for item in pair[1]]
        header_content_pairs[i] = (pair[0], adjusted_contents)

    header_content_pairs = handle_overlaps(header_content_pairs)    
    header_content_pairs = filter_content(header_content_pairs)
    return header_content_pairs

def recursive_text_extraction(element):
    """ Recursively extract text from any element, including nested structures """
    text_parts = []
    for child in element.children:
        if isinstance(child, NavigableString):
            text = str(child).replace('\xa0', ' ').strip()  # Replace non-breaking spaces with regular spaces
            if text:
                text_parts.append(text)
        elif child.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']:
            text_parts.append(recursive_text_extraction(child))

    return " ".join(text_parts)

def extract_content_from_cell(cell):
    """ Extracts the content from an html table cell """
    content = []
    if cell.find('p'):
        paragraphs = cell.find_all('p')
        for p in paragraphs:
            text = recursive_text_extraction(p)
            content.append(text)
    elif cell.find(['ul', 'ol']):
        items = cell.find_all('li')
        for li in items:
            item_text = ' '.join(text for text in li.stripped_strings)
            content.append(item_text)
    else:
        content.append(cell.get_text(" ", strip=True))


    return '\n'.join(content)  # Join contents with newline for clarity

def process_table_row(row, column_headers):
    """ Extracts the content from an html table row """
    cells = row.find_all('td')
    row_data = []
     # Check if there's a <th> tag in this row, use it as the row header
    row_header = row.find('th')
    row_header_text = row_header.get_text(strip=True) if row_header else None

    # Use first cell as row header if no <th> and no column headers
    if not column_headers and not row_header_text and len(cells) == 2:
        row_header_text = extract_content_from_cell(cells[0])
        cells = cells[1:]  # Skip the first cell as it's used as header

    # Process each cell based on the presence of column headers
    for idx, cell in enumerate(cells):
        if column_headers:
            header_index = idx if idx < len(column_headers) else -1  # Ensure no out-of-range index
            if row_header_text:
                cell_header = f"{row_header_text} : {column_headers[header_index+1]}" if (row_header and (header_index + 1 < len(column_headers))) else f"{row_header_text} : {column_headers[header_index]}"
            else: 
                cell_header = column_headers[header_index]
        else:
            cell_header = row_header_text if row_header_text else ""

        cell_content = extract_content_from_cell(cell)
        if cell_header and cell_content:
            row_data.append(f'{cell_header} : {cell_content}')

    return (('tr', ' \n'.join(row_data)))

def process_table_data(table):
    """ Extracts the content from an html table """
    body_rows = table.find('tbody').find_all('tr') if table.find('tbody') else table.find_all('tr')
    column_headers = []

    thead = table.find('thead')
    if thead:
        th_elements = thead.find_all('th') or thead.find_all('td')  # Handling case if 'th' elements are absent.
        column_headers = [th.get_text(strip=True) for th in th_elements]

    if not column_headers:  # Check within first 'tr' of 'tbody' if 'thead' was insufficient or absent.
        first_tr = table.find('tbody').find('tr') if table.find('tbody') else table.find('tr')
        if first_tr and len(first_tr.find_all('th')) > 1:
            column_headers = [th.get_text(strip=True) for th in first_tr.find_all('th')]
            body_rows = body_rows[1:]  # Skipping header row

    all_data = []
    for row in body_rows:
        row_data = process_table_row(row, column_headers)
        all_data.append(row_data)

    return all_data

def process_description_list(dl):
    """ Extracts the content from an html description list """
    entries = []
    current_dts = []  # List to accumulate dt texts
    element = dl.find(['dt', 'dd'])  # Start with the first dt or dd
    
    while element:
        if element.name == 'dt':
            current_dts.append(element.get_text(strip=True))  # Collect dt text
        elif element.name == 'dd':
            if current_dts:  # Check if there are dt texts collected
                # Join all dt texts and append the current dd text
                entries.append(('dl', f"{' / '.join(current_dts)}: {element.get_text(strip=True)}"))
                current_dts = []  # Reset dt texts for next iteration
        element = element.find_next(['dt', 'dd'])  # Move to the next dt or dd
    return entries 

def extract_list_items(list_element):
    """ Extracts the content from an html list """
    list_content = []
    for li in list_element.find_all('li'):
        header_in_li = li.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if header_in_li:
            item_text = f"{header_in_li.text.strip()}: " + ' '.join(text for text in li.stripped_strings if text != header_in_li.text)
        else:
            item_text = ' '.join(text for text in li.stripped_strings)
        list_content.append(('li', item_text))
    return list_content

def calculate_depth(element, current_depth=0):
    """
    Calculates the depth of a given HTML element within the DOM tree.

    This function recursively traverses the DOM tree, calculating the depth of the given element by checking 
    the number of nested child elements.

    Parameters:
        element (Tag): A BeautifulSoup Tag object representing the element whose depth is to be calculated.
        current_depth (int): The current depth in the recursive traversal.

    Returns:
        int: The depth of the element within the DOM tree.
    """
    if not hasattr(element, 'contents') or not element.contents:
        return current_depth
    else:
        # Collect depths for all tag children
        depths = [calculate_depth(child, current_depth + 1) for child in element if isinstance(child, Tag)]
        # Return the maximum of the collected depths if not empty, otherwise return current depth
        return max(depths) if depths else current_depth

def process_paragraphs_and_lists(element, content):
    """ Extracts the content from paragraph and list hmtl tags """
    if element.name == 'p': #and not any(child for child in element.children if child.name):
        if element.contents:
            text = recursive_text_extraction(element)
            content.append(('p', text))
        else:
            content.append(('p', ' '.join(text.rstrip() for text in element.strings)))
            
    elif element.name in ['ul', 'ol']:
        list_items = extract_list_items(element)
        content.extend(list_items)
    elif element.name == 'dl':
        list_items = process_description_list(element)
        content.extend(list_items)
    return content

def process_content(content_list, anc_contents):
    """
    Processes content by removing unwanted spaces, segmenting text, and matching it with reference content.

    This function cleans and processes a list of content items by removing extra spaces, splitting text 
    into segments, and matching those segments against reference content (ancestor contents) to identify 
    overlaps or relevant segments.

    Parameters:
        content_list (list of tuples): A list of content items, where each item is a tuple (tag, text).
        anc_contents (list of str): A list of reference content strings used for matching.

    Returns:
        list of tuples: A list of processed content pairs that have been cleaned and matched against references.
    """
    space_pattern = r'\u00A0|\u200B|\u2002|\u2003|\u2004|\xa0|\u200b'
    pattern = r'\s{2,}(?!\s|$|[a-z])'

    content_list = [(item[0], re.sub(space_pattern, '', item[1]).strip()) for item in content_list]
    content_list = [item for item in content_list if item[1]!= '']

    if not any(item[1][0] in ['span', 'u'] for item in content_list if item):
        if not any (item[1][0].islower() for item in content_list if item):
            return content_list

    content_pairs = []
    content_texts = [item[1] for item in content_list]
    added_texts = []
    for item in content_list:
        found_in_ref = False
        segments = re.split(pattern, item[1])

        for segment in segments:
            segment = re.sub(space_pattern, '', segment).strip()
            if bool(re.search(r'[a-zA-Z]', segment)):# and segment not in content_texts:
                for ref in anc_contents:
                    if segment in ref:
                        found_in_ref = True
                        if ref not in added_texts:
                            content_pairs.append((item[0], ref))
                            added_texts.append(ref)
                        break

                if segment in content_texts and not found_in_ref:
                    seg_item = content_list[content_texts.index(segment)]
                    if seg_item not in content_pairs and seg_item[1] not in added_texts:
                        content_pairs.append(content_list[content_texts.index(segment)])
                        added_texts.append(content_list[content_texts.index(segment)][1])

    content_texts = [content[1] for content in content_pairs]
    for ref in anc_contents:
        ref = re.sub(space_pattern, '', ref).strip()
        if bool(re.search(r'[a-zA-Z]', ref)) and ref not in content_texts:
            content_pairs.append(('u', ref))
    
    return content_pairs

def process_content_item(item):
    """
    Processes a single content item by removing unwanted spaces and segmenting text.

    This function splits a single content string into segments based on a pattern and removes unnecessary 
    whitespace or special characters.

    Parameters:
        item (str): The content string to be processed.

    Returns:
        list: A list of cleaned and segmented text strings.
    """
    space_pattern = r'\u00A0|\u200B|\u2002|\u2003|\u2004|\xa0|\u200b'
    pattern = r'\s{2,}(?!\s|$|[a-z])'
    content = []
    segments = re.split(pattern, item)

    for segment in segments:
        segment = re.sub(space_pattern, '', segment).strip()
        if bool(re.search(r'[a-zA-Z]', segment)):
            content.append(segment)
    return content

def collect_div_content(div):
    """
    Recursively collects text content from a given div element until a header is encountered.

    This function traverses through a div and its child elements, extracting relevant text content
    from paragraphs, lists, and tables. It stops when encountering a header tag (h1-h6) and 
    returns the collected content.

    Parameters:
        div (Tag): A BeautifulSoup Tag object representing the div to collect content from.

    Returns:
        tuple: A tuple containing the collected div content (as a list) and a boolean indicating whether 
               a header was encountered.
    """
    div_content = []
    ref_content = process_content_item(recursive_text_extraction(div))
    found_header = False

    for child in div.children:
        if isinstance(child, NavigableString):
            text_content = child.get_text(strip=True)
            if text_content:
                div_content.append(('str', text_content))
            continue

        if child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            found_header = True
            break
        if child.name in ['p', 'ul', 'ol', 'dl']:
            div_content = process_paragraphs_and_lists(child, div_content)
        if child.name in ['span', 'strong', 'em', 'a', 'b']:
            nested_ps = child.find_all('p')
            if nested_ps:
                for p in nested_ps:
                    div_content = process_paragraphs_and_lists(p, div_content)
            else:
                text_content = recursive_text_extraction(child)
                if text_content:
                    div_content.append((child.name, text_content))
        if child.name == 'table':
            text_content = process_table_data(child)
            if text_content:
                div_content.extend(text_content)
        elif child.name in ['div', 'section']  or (child.contents):
            child_content, child_found_header = collect_div_content(child)
            div_content.extend(child_content)

            if child_found_header:
                found_header = True
                break
    if(ref_content):
        if not (div_content):
            div_content = [('u', ref) for ref in ref_content]
        else:
            div_content = process_content(div_content, ref_content)

    return div_content, found_header

def find_next(current_element):
    """ Finds the next element in the DOM tree to explore, never moving up in the tree structure """
    if not (current_element.find_next_sibling()):
        stop_search = False
        while not (stop_search):
            if current_element.parent:
                current_element = current_element.parent
            else:
                stop_search = True
            if(current_element.find_next_sibling()):
                stop_search = True

    current_element = current_element.find_next_sibling()
    
    return current_element

def collect_content(header):
    """
    Collects content related to a given header element by iterating through its siblings.

    This function processes HTML content by traversing the elements following a header tag, 
    collecting content from paragraphs, lists, tables, divs, and other relevant elements. It 
    stops collecting if a new header of the same or higher level is encountered or after 20 
    content items have been collected.

    Parameters:
        header (Tag): A BeautifulSoup Tag object representing the header from which content is collected.

    Returns:
        list of tuples: A list of collected content, where each item is a tuple (tag, content).
    """

    content = []
    current_element = find_next(header)

    while current_element and len(content)<20:
        if isinstance(current_element, NavigableString):
            current_element = current_element.next_sibling
            continue
        # Stop if a new header within the same or higher level is encountered
        if current_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            break
        # Process the element based on its type
        if current_element.name in ['p', 'ul', 'ol', 'dl']:
            content = process_paragraphs_and_lists(current_element, content)
        if current_element.name in ['span', 'strong', 'em', 'a', 'b']:
            nested_ps = current_element.find_all('p')
            if nested_ps:
                for p in nested_ps:
                    content = process_paragraphs_and_lists(p, content)
            else:
                text_content = recursive_text_extraction(current_element)
                if text_content:
                    content.append((current_element.name, text_content))
        if current_element.name == 'table':
            text_content = process_table_data(current_element)
            if text_content:
                content.extend(text_content)
        elif current_element.name in ['div', 'section'] or (calculate_depth(current_element) > 2):
            # Recursively collect text from divs or sections
            div_content, found_header = collect_div_content(current_element)
            content.extend(div_content)

            if found_header:
                break  # Stop if a header is found within the div
            # Move to the next sibling or up to a higher level if needed
        current_element = find_next(current_element)
    return content

def preprocess_soup(soup):
    """
    Preprocesses the BeautifulSoup object by removing unwanted headers, footers, and line breaks.

    This function modifies the HTML structure to remove unnecessary headers, footer sections, 
    and replaces <br> tags with newline characters to clean up the document before processing.

    Parameters:
        soup (BeautifulSoup): A BeautifulSoup object representing the parsed HTML document.

    Returns:
        None: The function modifies the soup object in-place.
    """
    # Find the footer tag
    header_tags = soup.find_all('header')
    footer_tag = soup.find('footer')
    
    if len(header_tags)==1:
        header_tags[0].decompose()
    # If footer tag exists
    if footer_tag:
        # Iterate over the siblings of the footer tag
        for sibling in footer_tag.find_next_siblings():
            # Remove each sibling
            sibling.decompose()

        # Remove the footer tag itself
        footer_tag.decompose()
    
    for br in soup.find_all("br"):
        br.replace_with("\n")

def organize_headers_content(soup):
    """
    Organizes content by associating it with the closest preceding header, then refines and filters it.

    This function processes the HTML by finding all headers, collecting the content that follows 
    each header, and organizing it into header-content pairs. It then applies additional filtering 
    and refinement to remove irrelevant or duplicate content.

    Parameters:
        soup (BeautifulSoup): A BeautifulSoup object representing the parsed HTML document.

    Returns:
        list of tuples: A list of header-content pairs after refinement and filtering.
    """
    preprocess_soup(soup)
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    headers = filter_elements(headers)

    valid_pairs = []
    last_valid_header = None

    for i, header in enumerate(headers):
        content = collect_content(header)
        content = [(item[0], item[1].strip()) for item in content if item[1].strip() != '']
        header_text = header.text.strip()

        if header_text == '' and content and (i!=(len(headers)-1)):
            if last_valid_header is not None:
                # Extend the content of the last valid header
                last_valid_header[1].extend(content)
            else:
                new_pair = (header_text, content)
                valid_pairs.append(new_pair)
        elif header_text != '':
            # Create a new valid pair and update last_valid_header
            new_pair = (header_text, content)
            valid_pairs.append(new_pair)
            last_valid_header = new_pair

    valid_pairs = refine_content(soup, valid_pairs)
    valid_pairs = filter_cookie_content(valid_pairs)

    return valid_pairs

# Define a custom aggregation function
def column_agg(series):
    """
    Custom aggregation function to handle the aggregation of columns based on their name.

    This function provides custom rules for aggregating different columns. For certain columns, 
    it returns a list of values, while for others it returns the first non-null value.

    Parameters:
        series (pandas.Series): A pandas Series representing a column of data to be aggregated.

    Returns:
        list or value: A list of values for 'URL', 'Clean_URL', and 'Title' columns, None for 'Previous_Header', 
        and the first non-null value for other columns.
    """
    if series.name in ['URL', 'Clean_URL', 'Title']:
        return list(series)
    elif series.name in ['Previous_Header']:
        return None
    else:
        # For other columns, return the first non-null value
        return series.dropna().iloc[0]