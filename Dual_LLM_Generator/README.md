# Dual LLM RAG pipeline (batch and online) with Vertex AI Search and Judge Model

This folder provides a pipeline for evaluating and comparing responses from two different Large Language Models (LLMs) - typically Google Gemini (via Vertex AI) and a local Ollama model. It leverages Retrieval-Augmented Generation (RAG) using Google Cloud Vertex AI Search and includes an optional LLM-based 'Judge' to automatically assess response quality based on retrieved context.

There are two scripts in this folder: one processes multiple questions read from a TSV input file and the other processes one question at a time. Both output detailed evaluation results in a structured file (CSV, JSON, or TSV, your choice).

## Features

* **Dual LLM Comparison:** Generates outputs (responses) from two distinct LLMs (e.g., Vertex AI Gemini vs. local Ollama) for the same user question.
* **Vertex AI Search RAG:** Retrieves RAG context using Google Cloud Vertex AI Search to augment generation (RAG).
* **Advanced Prompting:** Utilizes configurable prompts loaded from files for query rephrasing and final answer summarization/generation.
* **LLM Judge:** Optionally employs a separate LLM (e.g., Gemini 2.0 Flash or Pro) to automatically evaluate and compare the generated responses based on the retrieved context.
* **Batch Processing:** Reads multiple questions from a TSV input file (`dataset.tsv` by default) or a single question. The dataset must have the following columns: case_id, 	query,	ground_truth_rephrased_query,		ground_truth_final_answer.
* **Online Processing:** Takes in a single question from the command line and returns the rephrased query and the final answer. 
* **Structured Output:** Returns detailed results, including interim and final responses, their timings, their token counts (where available), errors, RAG derived context, and judge model verdicts, to a CSV, JSON, or TSV file.
* **Command-Line Interface:** Configurable execution via command-line arguments.

## File Structure

1. There are two main scripts: dual_llm_rag_online.py and dual_llm_rag_batch.py

* `dual_llm_rag_online.py`: The main Python script to execute the entire pipeline for a single query. *(This script is helpful to compare answers for a single question)*
* `requirements.txt`: Lists the required Python packages for installation.
* `prompts/`: Directory containing essential prompt template files:
    * `system_instruction.txt`: Base system instructions applied to LLMs.
    * `rephraser.txt`: Prompt template used for query rephrasing.
    * `summarizer.txt`: Prompt template used for generating the final answer (using context if RAG is enabled).
    * `judge_prompt.txt`: System instructions for the Judge LLM (if used).
    * `judge_model_name.txt`: File containing the specific Vertex AI model name to use as the Judge  (e.g., `gemini-2.0-flash-001`).
* `-q` (query: "Give me a spicy fish recipe"): A string containing a question that you want to be processed. 
* `evaluation_results.csv` (Default Output): The file where the script saves the detailed evaluation results. The format (CSV, JSON, TSV) depends on the extension specified in the `--output-path` argument.

* `dual_llm_rag_batch.py`: The main Python script to execute the entire pipeline for multiple queries in a TSV or CSV file. *(This script is helpful to compare answers for multiple questions)*
* `-i` (`.path_to_file/dataset.tsv`) (Default Input): A Tab-Separated Value file containing the questions to be processed. Must include a header row with a column named `case_id`, 	`query`, `ground_truth_rephrased_query`,		`ground_truth_final_answer`.
* `requirements.txt`: Lists the required Python packages for installation.
* `prompts/`: Directory containing essential prompt template files:
    * `system_instruction.txt`: Base system instructions applied to LLMs.
    * `rephraser.txt`: Prompt template used for query rephrasing.
    * `summarizer.txt`: Prompt template used for generating the final answer (using context if RAG is enabled).
    * `judge_prompt.txt`: System instructions for the Judge LLM (if used).
    * `judge_model_name.txt`: File containing the specific Vertex AI model name to use as the Judge (e.g., `gemini-2.0-flash-001`).
* `evaluation_results.csv` (Default Output): The file where the script saves the detailed evaluation results. The format (CSV, JSON, TSV) depends on the extension specified in the `--output-path` argument.


## Prerequisites

* Python 3.8+
* pip (Python package installer)
* Git (for cloning the repository)
* **Google Cloud Account:** With billing enabled.
* **Google Cloud Project:** A designated project ID.
* **Enabled Google Cloud APIs:**
    * Vertex AI API
    * Vertex AI Search API (Required only if using the RAG feature)
* **Google Cloud Storage (GCS):** A bucket to store documents for RAG (Required only if using RAG).
* **Vertex AI Search:** A configured Datastore linked to your GCS bucket (Required only if using RAG). Make note of your **Datastore ID**.
* **Ollama:** (Optional, if comparing against a local model)
    * Install Ollama locally ([https://ollama.com/](https://ollama.com/)).
    * Pull the models you intend to use (e.g., `ollama pull llama3`).
    * Ensure the Ollama service is running when executing the script.
* **Google Cloud SDK (`gcloud`):** Installed and configured locally.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Create and activate a virtual environment:** (Highly Recommended)
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # OR
    # .venv\Scripts\activate  # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Authenticate with Google Cloud:** Use Application Default Credentials (ADC).
    ```bash
    gcloud auth application-default login
    ```
5.  **Set Default Project (Optional but Recommended):**
    ```bash
    export PROJECT_ID="YOUR_PROJECT_ID"
    export LOCATION="us-central1" (or any other region)
    export GOOGLE_VERTEXAI="True" (if you are using models from Google or RAG from Google Vertex Search)
    ```
6.  **(If using RAG) Prepare Cloud Resources:**
    * Create your GCS bucket.
    * Upload your source documents (e.g., PDFs from the `recipes/` folder if applicable) to the GCS bucket.
    * Create and configure your Vertex AI Search Datastore, linking it to the GCS bucket. Allow time for indexing. Note the **Datastore ID**.
7.  **(If using Ollama) Prepare Ollama:**
    * Ensure the Ollama application/service is running.
    * Pull the required models (e.g., `ollama pull llama3`).

## Configuration

* **Models:** The script uses default models (Gemini 2.0 Flash, Ollama Llama3). To change these, modify the `DEFAULT_LEFT_CONFIG` and `DEFAULT_RIGHT_CONFIG` dictionaries directly within the Python script (`run_eval_no_ui.py`).
* **Prompts:** Customize the behavior of the rephraser, summarizer, and judge by editing the corresponding text files in the `prompts/` directory. Ensure `judge_model_name.txt` contains a valid and accessible Vertex AI model name.
* **Vertex AI Search (RAG):** Enable RAG using the `--use-rag` flag. You MUST also provide `--project-id`, `--location`, and `--datastore-id` via command-line arguments (or ensure the project/location environment variables are set). Ensure your credentials have permission to access these resources.
* **Judge Model:** Enable the judge using the `--judge` flag. Ensure the model specified in `prompts/judge_model_name.txt` is available in your Vertex AI project.

## Usage

Run the script from your terminal within the activated virtual environment.

Example (No RAG, No Judge - quick option with single question):

```bash
python rag_demo_dual_llms_single_query.py -q "Can you give me a spicy fish recipe?" -o results_no_rag.csv
```

Example (No RAG, No Judge - quick option with a CSV or TSV file ):
```bash
python dual_llm_rag_batch.py \
    -i "dataset.tsv" \
    -o results_with_rag.csv
```


Example (With RAG, No Judge - this takes in a CSV or TSV file ):
```bash
python dual_llm_rag_batch.py \
    -i "dataset.tsv" \
    --use-rag \
    --project-id "XXX-123" \
    --location "us-central1" \
    --datastore-id "YYY-123" \
    --datastore-region "global" \
    -o results_with_rag.csv
```

Example (With RAG, No Judge - quick option with single question ):
```bash
python dual_llm_rag_online.py \
    -q "Give me a spicy fish recipe" \
    --use-rag \
    --project-id "XXX-123" \
    --location "us-central1" \
    --datastore-id "YYY-123" \
    --datastore-region "global" \
    -o results_with_rag.csv
```

Example (With RAG and Judge) - this takes in a CSV file:
```bash
python dual_llm_rag_batch.py \
    -i "dataset.tsv" \
    --use-rag \
    --project-id "XXX-123" \
    --location "us-central1" \
    --datastore-id "YYY-123" \
    --datastore-region "global" \
    -o results_with_rag.csv
```

Example (With RAG and Judge) quick option with single question:
```bash
python dual_llm_rag_online.py \
    -q "Give me a spicy fish recipe" \
    --use-rag \
    --project-id "XXX-123" \
    --location "us-central1" \
    --datastore-id "YYY-123" \
    --datastore-region "global" \
    --judge \
    -o results_rag_judge.json
```

