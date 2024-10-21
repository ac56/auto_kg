# Automatic Knowledge Graph Generation

This project includes several Python scripts that depend on functions organized into separate directories. To ensure that the scripts can locate and import functions from these directories, you need to set the `PYTHONPATH` environment variable.

This `README.md` will guide you through the steps to set the `PYTHONPATH` on your system to point to the project root directory.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setting Up the PYTHONPATH](#setting-up-the-pythonpath)
    - [Windows](#windows)
    - [Linux / macOS](#linux--macos)
3. [How to Run the Scripts](#how-to-run-the-scripts)
4. [Configuration](#configuration)

---

## Project Structure

The directory structure for this project is as follows:

```plaintext
auto_kg/
│
├── bert_classification/
│   └── feature_classifier/
|       └── results/
|       └── final_model.zip
|       └── final_model_tokenizer.zip
│   └── main_scripts/
|       └── feature_classifier_inference.py
|       └── feature_classifier_training.py
|       └── match_classifier_inference.py
|       └── match_classifier_training.py
│   └── match_classifier/
|       └── results/
|       └── final_model.zip
|       └── final_model_tokenizer.zip
|
├── cluster_data/
│
├── dbpedia/
│
├── few-shot_prompts/
│   └── feature_prompts/
|       └── prompt_1.txt
|       └── ...
|       └── prompt_8.txt
│   └── match_prompts/ 
|       └── match_prompts/
|       └── prompt_1.txt
|       └── prompt_2.txt
|       └── prompt_3.txt
|
├── function_files/
│   └── __init__.py
│   └── bert_functions.py
│   └── classifier_functions.py
│   └── clustering_functions.py
│   └── content_functions.py
│   └── crawler_functions.py
│   └── dataset_functions.py
│   └── dbpedia_functions.py
|   └── kg_functions.py
|   └── phrase_functions.py
|
├── init_data/
│   └── init_neo4j/
|       └── feat_cluster_data.csv
|       └── feat_uri_data.csv
|       └── hierarchy.csv
|       └── nodes.csv
|       └── uri_data
|       └── uri_data.csv
│   └── concepts.csv
│   └── vendor_urls.csv
|
├── main_scripts/
│   └── __init__.py
│   └── cluster_resource_matching.py
│   └── clustering.py
│   └── crawler_main.py
│   └── dataset_preparation.py
│   └── dbpedia_data_collection.py
│   └── feature_classification.py
│   └── html_processing.py
│   └── kg_data_preparation.py
│   └── kg_generation.py
│   └── match_classification.py
│   └── phrase_extraction.py
|
├── neo4j/
│   └── updated/
|
├── web_data/
|
├── config.json
|
├── README.md
|
└── requirements.txt

```


## Setting Up the PYTHONPATH

To ensure that the scripts can locate the necessary modules from the `auto_kg` directory, you need to set the `PYTHONPATH` environment variable to point to the project root.

### Windows

1. Open the **Start Menu** and search for **Environment Variables**.
2. Click on **Edit the system environment variables**.
3. In the **System Properties** window, click on the **Environment Variables** button.
4. Under **System variables**, scroll down and find `PYTHONPATH`. If it doesn’t exist, click **New** and set:
   - **Variable name**: `PYTHONPATH`
   - **Variable value**: The full path to your `auto_kg` project directory.
Example: `C:\path\to\auto_kg`
5. Click OK to close all windows.
6. Restart any open command prompt or terminal windows to apply the changes.


### Linux / macOS

1. Open a terminal window.
2. Open your shell configuration file (e.g., .bashrc, .zshrc, or .bash_profile), depending on your shell:
```bash
    nano ~/.bashrc  # For bash users
    nano ~/.zshrc   # For zsh users 
```
3. Add the following line to set the `PYTHONPATH`:
```bash
    export PYTHONPATH="/path/to/auto_kg:$PYTHONPATH"
```
Replace `/path/to/auto_kg` with the full path to your project directory.

4. Save the file and exit the editor.
5. Reload your shell configuration:
```bash
    source ~/.bashrc  # or ~/.zshrc
```
6. Verify that `PYTHONPATH` is set:
```bash
    echo $PYTHONPATH
```
There are two main ways to manage the Python environment for this project: using Conda or Virtualenv. Below are instructions for both.

### Using Conda

1. Create a new Conda environment:

```bash
    conda create --name auto_kg python=3.9
```
2. Activate the environment:

```bash
    conda activate auto_kg
```

3. Install the required dependencies:

```bash
    pip install -r requirements.txt
```
4. Verify the installation: Ensure that all required packages are installed successfully by running:   

```bash
    conda list
```

### Using Virtualenv

1. Create a new virtual environment:

```bash
    python -m venv env
```
2. Activate the virtual environment:

- Windows:

```bash
    .\env\Scripts\activate
```

- Linux/ MacOS:

```bash
    source env/bin/activate
```

3. Install the required dependencies:

```bash
    pip install -r requirements.txt
```
4. Verify the installation: Ensure that all required packages are installed successfully by running:   

```bash
    pip list
```

## How to Run the Scripts

Once the environment is set up and the `PYTHONPATH` is correctly configured, you can run the Python scripts.

1. Activate the environment (if not already activated):

- Conda:

```bash
    conda activate auto_kg
```
- Virtualenv:

```bash
    source env/bin/activate  # or .\env\Scripts\activate for Windows
```
2. Run a script: For example, to run the feature classification using BERT:

```bash
    python bert_classification/main_scripts/feature_classifier_inference.py
```

3. By running the scripts, some of the folders (e.g. `cluster_data` and `dbpedia`) will be populated with the results of different parts of the pipeline. It is important to follow the order in which the scripts should be run, as this will ensure the relevant input to each stage is available. All the scripts executing the main pipeline are in the `auto_kg/main_scripts/` folder. 

### Order of Script Execution:

1. **`crawler_main.py`** - Collects web data from vendor websites to populate the `web_data` directory.
2. **`html_processing.py`** - Processes the collected HTML files and extracts relevant information to populate intermediate files.
3. **`dbpedia_data_collection.py`** - Collects data from DBpedia and stores it in the `dbpedia` folder.
4. **`phrase_extraction.py`** - Extracts key phrases from the processed data and prepares it for further steps. 
5. **`clustering.py`** - Clusters the extracted phrases and stores the results in the `cluster_data` folder.
6. **`cluster_resource_matching.py`** - Applies a preliminary match between the phrase clusters and relevant resources from DBpedia.
7. **`dataset_preparation.py`** - Prepares the dataset for the 2 classification tasks: feature classification and match classification. The data for these tasks is stored in the `datasets` folder.
8. **`feature_classification.py`** - Classifies features using the Gemini model, the result is stored again in the `datasets` folder.
9. **`match_classification.py`** - Classifies if each of the phrase clusters matches the collected DBpedia resources paired to it, storing the result in the `datasets` folder.
10. **`kg_data_preparation.py`** - Prepares the data for knowledge graph generation by organizing the classified feature data and the relations.
11. **`kg_generation.py`** - Generates the final knowledge graph (including the hierarchy of nodes) based on the processed and classified data. 

### Notes

- The **`feature_classification.py`** and **`match_classification.py`** scripts are using Google's Gemini language model through the Google Cloud API. This will need additional setup of the Vertex AI Studio, instructions for which are found [here](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal). After the setup is complete, a system environment variable should be set named **`GOOGLE_API_KEY`**, to provide your API key to the system. The key is made available through the [Google AI Studio](https://aistudio.google.com/app/apikey)

- Alternatively, the corresponding scripts found in `auto_kg/bert_classification/main_scripts`, namely **`feature_classifier_inference.py`** and **`match_classifier_inference.py`** can be used for the same stage of the pipeline, which utilize trained distilBERT models. First, the folders `final_model` and `final_model_tokenizer` need to be unzipped and stored in the `bert_classification/feature_classifier` and `bert_classification/match_classifier` folders, respectively. If the classifiers are not yet in those locations, the can be downloaded from [this](https://doi.org/10.5281/zenodo.13899562) Zenodo repository and should be extracted in the specified directories. Note that unzippining a folder creates a folder with the same name in the specified path and the specified path should be `auto_kg/bert_classification/{feature/match}_classifier`. After the code, executes, a file named **`feature_classified`** or **`match_classified`** should be stored in the `auto_kg/bert_classification/results/` folder, which should be copied and pasted in the `auto_kg/datasets/` folder for further steps.

- For the knowledge graph generation, there are some initial files that need to be used, which define the initial set of nodes and hierarchy based on the [NIST Cloud Computing Reference Architecture](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication500-292.pdf). After all previous steps are complete, a csv file named **`init_dataset.csv`** should be present in the **`datasets/`** folder. That file can be explored for information about the formed phrase clusters. The initial files in `auto_kg/init_data/init_neo4j/` define the blueprint of the knowledge graph in terms of nodes and hierarchy, then these nodes along with new ones will be populated with information extracted from each vendor's website. The CSV file `auto_kg/init_data/init_neo4j/feat_cluster_data.csv` should be updated using `init_dataset.csv` by looking for the initial set of features in the `Derived Phrase` column and listing down the clusters associated with each of these features (one cluster per row). After the knowledge graph generation scripts are executed, the resultant files defining the final knowledge graph are stored in `auto_kg/neo4j/` folder, ready for import into an [AuraDB database instance](https://console.neo4j.io/).


## Configuration

The config.json file contains various settings used in the project. Below is an explanation of each key in the file:

- `include_terms`: A list of terms to include when filtering URLs based on their path. These terms are typically related to cloud computing and relevant features.
- `exclude_terms`: A list of terms that should be excluded from the selected URLs (e.g., links to resources like blogs, manuals, etc.).
- `excluded_extensions`: File extensions that should be excluded from scraping or processing (e.g., images and non-text files like .pdf or .jpg).
- `repeat_limit`: The maximum number of times a vendor's content can appear.
- `vendor_name_dict`: A dictionary mapping vendor names to various aliases. For example, "Microsoft Azure" could be referred to as either "Microsoft" or "Azure".
- `embedding_model`: The pre-trained sentence embedding model used for processing textual data (for phrase embeddings)
- `pca_dim`: The number of dimensions for PCA dimensionality reduction of embeddings.
- `min_cluster_size`: The minimum size of a cluster for it to be considered valid.
- `min_samples`: The minimum number of samples required to form a cluster.
- `n_min` and `n_max`: The minimum and maximum number of n-grams to generate for cluster aggregation (derive name of each cluster)
- `min_freq`: The minimum frequency of an n-gram to be included in the output.
- `sim_threshold`: The similarity threshold for determining if two items are related.
- `n_abst_sent`: Number of abstract sentences to use in the match classification task.
- `n_contexts`: Number of context sentences (from web sources) to use for classification.
- `gemini_model_name`: The model used for Gemini language processing (e.g., gemini-1.5-pro).
- `match_prompt_select`: The prompt example to select for match classification (3 selects `auto_kg\few-shot_prompts\match_prompts\prompt_3.txt`).
- `feature_prompt_select`: Number of feature classification prompt examples to select (8 selects `auto_kg\few-shot_prompts\feature_prompts\prompt_8.txt`).
- `bert_feature_model_name`: The BERT model name for feature classification tasks.
- `bert_match_model_name`: The BERT model name for match classification tasks.


