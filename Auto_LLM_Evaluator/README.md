# Auto Alignment of LLM Evaluation Framework 
Authors: Jennifer Liang and Ram Seshadri

A configurable Python framework for evaluating Large Language Model (LLM) responses against ground truth data, initially designed for recipe/instruction-based use cases but adaptable via configuration.

This tool implements a multi-stage evaluation process:
1.  **Rephraser Evaluation:** Compares semantic similarity of rephrased queries (e.g., user input vs. LLM's understanding).
2.  **Final Answer Evaluation:**
    * **Source Score:** Compares identified sources (e.g., "Recipe 1", "Document A") cited in the LLM response vs. ground truth.
    * **Ingredient Score:** Compares the semantic similarity of extracted ingredient lists.
    * **Instruction Score:** Compares the semantic similarity of extracted instruction steps.

Scores are calculated based on semantic similarity and exact matches, with configurable penalties for incorrect or missing elements in the LLM response compared to the ground truth.

## Features

* **Configurable:** Customize column names, regex patterns, similarity thresholds, scoring weights, and penalties via a `config.yaml` file.
* **Vertex AI Integration:** Uses Google Cloud Vertex AI Text Embedding models for semantic similarity calculations.
* **Modular Design:** Encapsulated in an `LLMEvaluator` class for easy integration.
* **Robust:** Includes error handling and logging.
* **Extensible:** Structure allows for adding new evaluation stages or metrics.

## Prerequisites

* Python 3.8+
* Google Cloud Project with Vertex AI API enabled.
* Authentication: Set up Google Cloud Application Default Credentials (ADC). Typically done via `gcloud auth application-default login`. Ensure the authenticated user/service account has the "Vertex AI User" role or equivalent permissions.
* google-cloud-aiplatform library
* sentence-transformers library

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd llm-evaluator
    ```

2.  **Install using pip:**
    It's highly recommended to use a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
    Install the package and its dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    To include development dependencies (like `pytest` for running tests):
    ```bash
    pip install .[dev]
    ```

## Configuration (`config.yaml`)

Before running, create a `config.yaml` file. See the example `config.yaml` provided in the repository (or in the code documentation) for the structure and available parameters.

**Key sections:**

* `embedding_model_id`: Specify the Vertex AI embedding model (e.g., `text-embedding-004`).
* `column_names`: Map your DataFrame's column names to the roles needed by the evaluator.
* `regex_patterns`: Define regular expressions for extracting sources, ingredients, instructions, and identifying ignorable sentences.
* `thresholds`: Set cosine similarity thresholds for matching rephrased queries, ingredients, instructions, and "No Answer" cases.
* `scoring`: Define max points per section and penalty multipliers.
* `no_answer_keywords`: List keywords used to detect "No Answer" responses.

*Note:* Ensure your Google Cloud Project ID and Location are either set as environment variables (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION`) or specified directly in the `config.yaml` (using `project_id` and `location` keys).

## Usage

```python
import pandas as pd
from llm_evaluator.evaluator import LLMEvaluator # Correct import path

# 1. Define path to your configuration file
CONFIG_PATH = "config.yaml"

# 2. Load your data into a Pandas DataFrame
# Make sure the DataFrame has the columns specified in config.yaml
try:
    # Example: Load from a TSV file
    input_df = pd.read_csv("your_data.tsv", sep='\t')
except FileNotFoundError:
    print("Error: Data file not found.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()


try:
    # 3. Initialize the evaluator
    evaluator = LLMEvaluator(config_path=CONFIG_PATH)

    # 4. Run the evaluation pipeline
    results_df = evaluator.evaluate(input_df)

    # 5. Process or display the results
    print("Evaluation complete. Results DataFrame head:")
    print(results_df.head())

    # Example: Display key scores
    key_scores = results_df[[
        evaluator.col_names['case_id'], # Access configured case_id column name
        'rephraser_score',
        'source_score',
        'sentences_score_ingredient',
        'sentences_score_instruction',
        'final_score'
    ]]
    print("\nKey Scores:")
    print(key_scores.to_string())

    # Example: Save results to a new file
    # results_df.to_csv("evaluation_output.tsv", sep='\t', index=False)
    # print("\nResults saved to evaluation_output.tsv")

except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
    print(f"Configuration or Data Error: {e}")
except Exception as e:
    # Catch potential errors during Vertex AI init or evaluation
    print(f"An error occurred during evaluation: {e}")
```
