import time

def load_prompt_template(file_path):
    """
    Loads a prompt template from a specified file path.

    Args:
        file_path (str): Path to the file containing the prompt template.

    Returns:
        str: The loaded prompt template as a string.
    """
    with open(file_path, 'r') as file:
        template = file.read()
    return template

def create_match_prompt(row, template):
    """
    Creates a prompt for the match classification task using the provided template and data row.

    This function takes a data row containing relevant information and injects it into a 
    predefined prompt template using string formatting (.format()). The resulting prompt 
    can then be used for a matching task, such as determining the relationship between 
    two entities.

    Args:
        row (dict): A dictionary containing data for the prompt, including:
            - Names (str): The first entity name.
            - Abstract Sent (str): A sentence from the abstract.
            - Derived Phrase (str): The second entity name.
            - Sel Contexts (list): A list of relevant context sentences.
        template (str): The prompt template string.

    Returns:
        str: The formatted prompt string ready for use.
    """
    term1 = row['Names']
    abst_sent = row['Abstract Sent']
    term2 = row['Derived Phrase']
    contexts = row['Sel Contexts']
    context_string = " / ".join(contexts)
    
    # Use .format() to inject variables into the template
    prompt = template.format(
        term1=term1,
        abst_sent=abst_sent,
        term2=term2,
        context_string=context_string
    )
    
    return prompt

def create_feature_prompt(row, template):
    """
    Creates a prompt for the feature classification task using the provided template and data row.

    Similar to create_match_prompt, this function takes a data row and injects its 
    information into a predefined template. However, this is designed for feature 
    extraction tasks, focusing on a single entity and its context.

    Args:
        row (dict): A dictionary containing data for the prompt, including:
            - Derived Phrase (str): The entity of interest.
            - Sel Contexts (list): A list of relevant context sentences.
        template (str): The prompt template string.

    Returns:
        str: The formatted prompt string ready for use.
    """
    term = row['Derived Phrase']
    contexts = row['Sel Contexts']
    context_string = " / ".join(contexts)

    prompt = template.format(
        term=term,
        context_string=context_string
    )
    
    return prompt

def get_gemini_response(row, model):
  """
  Retrieves the response from Gemini for a given prompt.

  This function first checks if a Gemini response is already stored in the data row. 
  If not, it constructs the prompt based on the information in the row and retrieves 
  a fresh response from the Gemini model. A short delay (10 seconds in this example) 
  is included to avoid overwhelming the model with requests.

  Args:
      row (dict): A dictionary containing data for the prompt, including:
          - Gemini Response (str, optional): Previously obtained response from Gemini.
          - Prompt (str): The prompt string to send to Gemini.
      model (object): The Gemini model object used for generating content.

  Returns:
      str: The text response retrieved from Gemini.
  """
  if row['Gemini Response'] is not None:
    return row['Gemini Response']
  prompt = row['Prompt']
  time.sleep(10)
#   print('Retrieving Gemini response...')
  response = model.generate_content(prompt)
  return response.text